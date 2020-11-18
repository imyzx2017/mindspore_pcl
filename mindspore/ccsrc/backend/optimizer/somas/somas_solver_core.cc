/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/optimizer/somas/somas_solver_alg.h"
#include "backend/optimizer/somas/somas_solver_core.h"
#include "backend/optimizer/somas/somas_solver_pre.h"

using std::sort;
using std::unordered_map;
using std::vector;

namespace mindspore {
namespace somas {
Status SomasSolverCore::MemoryAllocationSolver() {
  auto start = std::chrono::system_clock::now();
  Status retval = SUCCESS;
  size_t best = SIZE_MAX;
  size_t best_timing = SIZE_MAX;
  if (all_) {  // loop over all heuristics
    FittingType best_branching = kBest;
    SortingType best_sorting = kGreaterSizeSmallerIndex;
    AlgorithmType best_algorithm = kManyObjects;
    uint32_t best_sol = 0;
    size_t worst = 0;
    BuildBlocks();
    Clean();
    MS_LOG(INFO) << "time\tSol#\tResult\t\t\t\tAlgorithm\tSorting Strategy\tOffset Strategy";
    for (size_t algorithm = 0; algorithm < kNumAlgorithmTypes; algorithm++) {
      algorithm_ = static_cast<AlgorithmType>(algorithm);
      for (size_t sort_strategy = 0; sort_strategy < kNumSortingTypes; sort_strategy++) {
        sort_strategy_ = static_cast<SortingType>(sort_strategy);
        SortTensors();
        for (size_t branching_strategy = 0; branching_strategy < kNumFittingTypes; branching_strategy++) {
          branching_strategy_ = static_cast<FittingType>(branching_strategy);
          Clean();
          MS_LOG(DEBUG) << "Timing Start " << tensors_.size() << " Tensors";
          start = std::chrono::system_clock::now();
          upperbound_ = FindSolutions();
          MS_LOG(DEBUG)
            << "\nElapsed time of upper bound testing: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
            << " ms";
          start = std::chrono::system_clock::now();

          if (upperbound_ > worst) {
            worst = upperbound_;
          }
          if (upperbound_ < best || upperbound_ == best) {
            best = upperbound_;
            best_algorithm = algorithm_;
            best_branching = branching_strategy_;
            best_sorting = sort_strategy_;
            best_sol = sol_count_;
            best_timing = timing_;
          }
          Verify();
          sol_count_++;
        }
      }
    }
    upperbound_ = best;
    auto end = std::chrono::system_clock::now();
    size_t total_time = std::chrono::duration_cast<std::chrono::milliseconds>((end - start)).count();
    const double giga = 1024. * 1024. * 1024.;
    const double cent = 100.;
    MS_LOG(INFO) << "SOMAS SOLVER RESUME:";
    MS_LOG(INFO) << "Best Solution:[" << 1 + best_sol << "/" << sol_count_ << "] ";
    MS_LOG(INFO) << "Best result:" << best << " Bytes " << (best) / (giga) << " GB ("
                 << (best - lifelongmemory_) / (giga) << " GB + " << lifelongmemory_ / (giga)
                 << " GB from lifelong tensors)";

    MS_LOG(INFO) << "Best timing:" << best_timing << " ms";
    MS_LOG(INFO) << "Best algorithm: " << algorithm_type_[best_algorithm].c_str();
    MS_LOG(INFO) << "Best sorting strategy: " << sorting_[best_sorting].c_str();
    MS_LOG(INFO) << "Best offset strategy: " << branching_[best_branching].c_str();
    MS_LOG(INFO) << "Time elapsed: " << total_time << " ms";
    MS_LOG(INFO) << "Spread:" << static_cast<double>((worst - best) / static_cast<double>(best * cent)) << " %%";
    best_sol_ = best_sol;
    SetBestSolution();
  } else {
    MS_LOG(INFO) << "Algorithm strategy: " << algorithm_type_[algorithm_].c_str();
    MS_LOG(INFO) << "Sorting strategy: " << sorting_[sort_strategy_].c_str();
    MS_LOG(INFO) << "Offset strategy: " << branching_[branching_strategy_].c_str();
    BuildBlocks();
    SortTensors();
    upperbound_ = FindSolutions();
    Verify();
  }
  return retval;
}

Status SomasSolverCore::Verify() {
  Status retval = SUCCESS;
  if (verify_) {
    MS_LOG(INFO) << "Verifying solution..";

    if (!Verify(upperbound_)) {
      MS_LOG(WARNING) << "Solver Allocation Memory Check FAILS";
      retval = FAILED;
    } else {
      const double giga = 1024. * 1024. * 1024.;
      MS_LOG(INFO) << "Solver Allocation Memory Check SUCCESS !!";
      MS_LOG(INFO) << "Result: " << upperbound_ << " (" << (upperbound_) / (giga) << " GB)";
      retval = SUCCESS;
    }
  }

  return retval;
}

Status SomasSolverCore::Verify(unordered_map<size_t, SomasSolverTensorDescPtr> *pTensor_map) {
  Status retval = SUCCESS;
  if (NULL == pTensor_map) return retval;
  MS_LOG(INFO) << "Verifying HQ Solution..";
  MS_LOG(INFO) << "Checking tensors id, sizes..";

  for (auto ptensor : *pTensor_map) {
    if (tensors_.count(ptensor.first) == 0) {
      MS_LOG(WARNING) << "HQ Tensor id " << ptensor.first << " does not exists";
    } else if (tensors_[ptensor.first]->size_ != ptensor.second->size_) {
      size_t HQ_index = ptensor.first;
      size_t HQ_size = ptensor.second->size_;
      size_t index = ptensor.first;
      size_t size = tensors_[ptensor.first]->size_;
      MS_LOG(WARNING) << "HQ Tensor Id: " << HQ_index << " with size: " << HQ_size
                      << " is different from Tensor Id: " << index << " size: " << size;
    }
  }

  MS_LOG(INFO) << "Checking HQ Solution..";
  tensors_ = *pTensor_map;
  retval = Verify(upperbound_) == 0 ? FAILED : SUCCESS;
  return retval;
}
bool SomasSolverCore::Verify(const size_t &upperbound) {
  auto start = std::chrono::system_clock::now();
  bool retval = true;
  size_t result = 0;
  SomasSolverTensorDescPtr t1;
  SomasSolverTensorDescPtr t2;

  for (auto t1_ : tensors_) {
    // check alignment
    result = std::max(result, t1_.second->size_ + t1_.second->offset_);
    for (auto t2_ : tensors_) {
      t1 = t1_.second;
      t2 = t2_.second;
      if (t1->index_ == t2->index_) continue;
      bool blifelong = (t1->lifelong_ || t2->lifelong_) && (t1->index_ != t2->index_);
      const size_t continuous = 2;
      const size_t conflict = 1;
      if ((*constraints_)(t1->index_, t2->index_) == continuous) {  // continuous constraint
        // t1 must be continous to t2
        bool bcontinuous = t1->offset_ == (t2->offset_ + t2->size_);
        if (!bcontinuous) {
          MS_LOG(WARNING) << "Continuous constraint violation in tensors " << t1->index_ << " and" << t2->index_;
          retval = false;
        }
      } else if (blifelong || (*constraints_)(t1->index_, t2->index_) == conflict) {  // conflict constraint
        size_t t1_ub = t1->offset_ + t1->size_;
        size_t t2_ub = t2->offset_ + t2->size_;
        bool b_overlap_lb = ((t2->offset_ >= t1->offset_) && (t2->offset_ < t1_ub));
        bool b_overlap_ub = ((t2_ub > t1->offset_) && (t2_ub < t1_ub));
        bool b_overlap = b_overlap_lb || b_overlap_ub;
        bool biszerosized = t1->size_ == 0 || t2->size_ == 0;
        if (b_overlap && !biszerosized) {
          MS_LOG(WARNING) << "Non-overlap constraint violation in tensors " << t1->index_ << " and" << t2->index_;
          retval = false;
        }
      }
    }
  }
  if (upperbound != result) {
    MS_LOG(WARNING) << "ERROR Invalid upperbound result --> Footprint Result: " << upperbound_
                    << " Tensor Result: " << result + lifelongmemory_;
    retval = false;
  }
  MS_LOG(DEBUG)
    << "\nElapsed time of Fast Heuristic Check: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << " ms";
  return retval;
}

void SomasSolverCore::BuildBlocks() {
  MS_LOG(DEBUG) << "Building block of tensors";

  lifelongmemory_ = 0;
  uint64_t tensors_block_count = 0;
  for (auto tensor : tensors_) {
    SomasSolverTensorDescPtr pTensor = tensor.second;
    if (pTensor->blocked_) continue;
    if (pTensor->lifelong_) {
      lifelongmemory_ += pTensor->size_;
      continue;
    }
    // move to the left
    while (pTensor->left_) pTensor = pTensor->left_;

    // set start tensor
    BlockTensor bTensor;
    bTensor.m_bre_allocate_ = true;
    bTensor.m_start_tensor_ = pTensor;
    // find size
    bTensor.m_size_ = 0;

    do {
      bTensor.m_size_ += pTensor->size_;
      pTensor->blocked_ = true;
      pTensor = pTensor->right_;
      tensors_block_count++;
    } while (NULL != pTensor);

    // add to the list
    this->block_tensors_.emplace_back(bTensor);
  }

  if (tensors_block_count != tensors_.size())
    MS_LOG(INFO) << static_cast<int>(tensors_.size() - tensors_block_count) << " lifelong tensors found";

  // for debug
  for (auto &b : block_tensors_) b.log();
}

void SomasSolverCore::Clean() {
  for (auto &block : block_tensors_) {
    block.m_bre_allocate_ = true;
    auto pTensor = block.m_start_tensor_;
    while (pTensor) {
      pTensor->offset_ = 0;
      pTensor = pTensor->right_;
    }
  }
  upperbound_ = SIZE_MAX;
}
void SomasSolverCore::SortTensors() {  // need to sort the tensors for Fast Heuristic
  MS_LOG(DEBUG) << "Sorting Blocks of tensor, strategy: " << sorting_[sort_strategy_].c_str();
  switch (sort_strategy_) {
    case kGreaterSizeSmallerIndex: {  // size(>), index(<)
      sort(block_tensors_.begin(), block_tensors_.end(), [](const BlockTensor &t1, const BlockTensor &t2) {
        return t1.m_size_ > t2.m_size_ ||
               (t1.m_size_ == t2.m_size_ && t1.m_start_tensor_->index_ < t2.m_start_tensor_->index_);
      });
      break;
    }
#ifdef SOMAS_DEBUG
    case kGreaterSizeGreaterIndex: {  // size(>), index(>)
      sort(block_tensors_.begin(), block_tensors_.end(), [](const BlockTensor &t1, const BlockTensor &t2) {
        return t1.m_size > t2.m_size ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->index_ > t2.m_pStartTensor->index_);
      });
      break;
    }
    case kGreaterSizeSmallerConstraintsSmallerIndex: {  // size(>), constraints(<), index(<)
      sort(block_tensors_.begin(), block_tensors_.end(), [](const BlockTensor &t1, const BlockTensor &t2) {
        return t1.m_size > t2.m_size ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->constraints_ < t2.m_pStartTensor->constraints_) ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->constraints_ == t2.m_pStartTensor->constraints_ &&
                t1.m_pStartTensor->index_ < t2.m_pStartTensor->index_);
      });
      break;
    }
    case kGreaterSizeSmallerConstraintsGreaterIndex: {  // size(>), constraints(<), index(>)
      sort(block_tensors_.begin(), block_tensors_.end(), [](const BlockTensor &t1, const BlockTensor &t2) {
        return t1.m_size > t2.m_size ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->constraints_ < t2.m_pStartTensor->constraints_) ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->constraints_ == t2.m_pStartTensor->constraints_ &&
                t1.m_pStartTensor->index_ > t2.m_pStartTensor->index_);
      });
      break;
    }
    case kGreaterSizeGreaterConstraintsSmallerIndex: {  // size(>), constraints(>), index(<)
      sort(block_tensors_.begin(), block_tensors_.end(), [](const BlockTensor &t1, const BlockTensor &t2) {
        return t1.m_size > t2.m_size ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->constraints_ > t2.m_pStartTensor->constraints_) ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->constraints_ == t2.m_pStartTensor->constraints_ &&
                t1.m_pStartTensor->index_ < t2.m_pStartTensor->index_);
      });
      break;
    }
    case kGreaterSizeGreaterConstraintsGreaterIndex: {  // // size(>), constraints(>), index(>)
      sort(block_tensors_.begin(), block_tensors_.end(), [](const BlockTensor &t1, const BlockTensor &t2) {
        return t1.m_size > t2.m_size ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->constraints_ > t2.m_pStartTensor->constraints_) ||
               (t1.m_size == t2.m_size && t1.m_pStartTensor->constraints_ == t2.m_pStartTensor->constraints_ &&
                t1.m_pStartTensor->index_ > t2.m_pStartTensor->index_);
      });
      break;
    }
#endif
    case kNumSortingTypes: {  // no sorting case
      break;
    }
  }
  // log for debug purposes
  for (auto &block : block_tensors_) block.log();
}

void SomasSolverCore::RestoreSolution(uint32_t sol_id) {
  for (auto block : block_tensors_) {
    if (block.offsets_.count(sol_id) == 0) assert(0);
    size_t bestOffset = block.offsets_[sol_id];
    size_t offset = bestOffset;
    SomasSolverTensorDescPtr pTensor = block.m_start_tensor_;

    while (pTensor) {
      pTensor->offset_ = offset;
      offset += pTensor->size_;
      pTensor = pTensor->right_;
    }
  }
}
size_t SomasSolverCore::Search(const std::shared_ptr<FootPrint> &pFootprint) {
  size_t result = 0;
  FastHeuristic fh;
  MS_LOG(INFO) << "Calling FastSolver Search for " << block_tensors_.size() << " tensors ";
  auto start = std::chrono::system_clock::now();
  if (fh.Eval(&block_tensors_, pFootprint, constraints_)) {
    result = pFootprint->Result();
    auto end = std::chrono::system_clock::now();
    timing_ = std::chrono::duration_cast<std::chrono::milliseconds>((end - start)).count();
    if (all_) {
      const double giga = 1073741824.;
      MS_LOG(INFO) << timing_ << " ms\t" << sol_count_ + 1 << "/"
                   << kNumFittingTypes * kNumAlgorithmTypes * kNumSortingTypes << "\t" << result << " Bytes ("
                   << result / giga << " GB)\t" << algorithm_type_[algorithm_].c_str() << "\t"
                   << sorting_[sort_strategy_].c_str() << "\t" << branching_[branching_strategy_].c_str();
    }
  } else {
    MS_LOG(INFO) << "FastSolver could not find solution";
  }

  if (result < upperbound_) {
    upperbound_ = result;
    best_sol_ = pFootprint->m_solId_;
    best_branching_ = branching_strategy_;
    best_sort_ = sort_strategy_;
  }

  return upperbound_;
}

void SomasSolverCore::AppendLifelongTensors() {
  MS_LOG(DEBUG) << "Appending lifelong tensors to solution";
  size_t offset = upperbound_;
  for (auto t_ : tensors_) {
    SomasSolverTensorDescPtr pTensor = t_.second;
    if (pTensor->lifelong_) {
      pTensor->offset_ = offset;
      offset += pTensor->size_;
    }
  }
  upperbound_ += lifelongmemory_;
  MS_LOG(DEBUG) << lifelongmemory_ << " bytes from lifelong tensors added to solution";
}

size_t SomasSolverCore::FindSolutions() {
  MS_LOG(DEBUG) << "Start allocating blocks,offset strategy: " << branching_[branching_strategy_].c_str();

  std::shared_ptr<FootPrint> pFootprint = std::make_shared<FootPrint>();
  pFootprint->setBranchingStrategy(branching_strategy_);
  pFootprint->setCurrentSol(sol_count_);
  pFootprint->setAlgorithm(algorithm_);
  Search(pFootprint);
  AppendLifelongTensors();
  Destroy(pFootprint);
  return upperbound_;
}

void SomasSolverCore::Destroy(std::shared_ptr<FootPrint> &pFootprint) {
  while (NULL != pFootprint) {
    if (NULL != pFootprint->Next()) {
      std::shared_ptr<FootPrint> &p = pFootprint;
      pFootprint = pFootprint->Next();
      // delete p;
      p = NULL;
    } else {
      // delete pFootprint;
      pFootprint = NULL;
    }
  }
}
}  // namespace somas
}  // namespace mindspore
