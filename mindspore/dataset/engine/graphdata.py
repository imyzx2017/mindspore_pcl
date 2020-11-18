# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
graphdata.py supports loading graph dataset for GNN network training,
and provides operations related to graph data.
"""
import atexit
import time
import numpy as np
from mindspore._c_dataengine import GraphDataClient
from mindspore._c_dataengine import GraphDataServer
from mindspore._c_dataengine import Tensor

from .validators import check_gnn_graphdata, check_gnn_get_all_nodes, check_gnn_get_all_edges, \
    check_gnn_get_nodes_from_edges, check_gnn_get_all_neighbors, check_gnn_get_sampled_neighbors, \
    check_gnn_get_neg_sampled_neighbors, check_gnn_get_node_feature, check_gnn_get_edge_feature, \
    check_gnn_random_walk


class GraphData:
    """
    Reads the graph dataset used for GNN training from the shared file and database.

    Args:
        dataset_file (str): One of file names in the dataset.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel
            (default=None).
        working_mode (str, optional): Set working mode, now supports 'local'/'client'/'server' (default='local').

            - 'local', used in non-distributed training scenarios.

            - 'client', used in distributed training scenarios. The client does not load data,
              but obtains data from the server.

            - 'server', used in distributed training scenarios. The server loads the data
              and is available to the client.

        hostname (str, optional): Hostname of the graph data server. This parameter is only valid when
            working_mode is set to 'client' or 'server' (default='127.0.0.1').
        port (int, optional): Port of the graph data server. The range is 1024-65535. This parameter is
            only valid when working_mode is set to 'client' or 'server' (default=50051).
        num_client (int, optional): Maximum number of clients expected to connect to the server. The server will
            allocate resources according to this parameter. This parameter is only valid when working_mode
            is set to 'server' (default=1).
        auto_shutdown (bool, optional): Valid when working_mode is set to 'server',
            when the number of connected clients reaches num_client and no client is being connected,
            the server automatically exits (default=True).

    Examples:
        >>> import mindspore.dataset as ds
        >>>
        >>> data_graph = ds.GraphData('dataset_file', 2)
        >>> nodes = data_graph.get_all_nodes(0)
        >>> features = data_graph.get_node_feature(nodes, [1])
    """

    @check_gnn_graphdata
    def __init__(self, dataset_file, num_parallel_workers=None, working_mode='local', hostname='127.0.0.1', port=50051,
                 num_client=1, auto_shutdown=True):
        self._dataset_file = dataset_file
        self._working_mode = working_mode
        if num_parallel_workers is None:
            num_parallel_workers = 1

        def stop():
            self._graph_data.stop()

        if working_mode in ['local', 'client']:
            self._graph_data = GraphDataClient(dataset_file, num_parallel_workers, working_mode, hostname, port)
            atexit.register(stop)

        if working_mode == 'server':
            self._graph_data = GraphDataServer(
                dataset_file, num_parallel_workers, hostname, port, num_client, auto_shutdown)
            atexit.register(stop)
            try:
                while self._graph_data.is_stoped() is not True:
                    time.sleep(1)
            except KeyboardInterrupt:
                raise Exception("Graph data server receives KeyboardInterrupt")

    @check_gnn_get_all_nodes
    def get_all_nodes(self, node_type):
        """
        Get all nodes in the graph.

        Args:
            node_type (int): Specify the type of node.

        Returns:
            numpy.ndarray: Array of nodes.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_nodes(0)

        Raises:
            TypeError: If `node_type` is not integer.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        return self._graph_data.get_all_nodes(node_type).as_array()

    @check_gnn_get_all_edges
    def get_all_edges(self, edge_type):
        """
        Get all edges in the graph.

        Args:
            edge_type (int): Specify the type of edge.

        Returns:
            numpy.ndarray: array of edges.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_edges(0)

        Raises:
            TypeError: If `edge_type` is not integer.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        return self._graph_data.get_all_edges(edge_type).as_array()

    @check_gnn_get_nodes_from_edges
    def get_nodes_from_edges(self, edge_list):
        """
        Get nodes from the edges.

        Args:
            edge_list (Union[list, numpy.ndarray]): The given list of edges.

        Returns:
            numpy.ndarray: Array of nodes.

        Raises:
            TypeError: If `edge_list` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        return self._graph_data.get_nodes_from_edges(edge_list).as_array()

    @check_gnn_get_all_neighbors
    def get_all_neighbors(self, node_list, neighbor_type):
        """
        Get `neighbor_type` neighbors of the nodes in `node_list`.

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neighbor_type (int): Specify the type of neighbor.

        Returns:
            numpy.ndarray: Array of nodes.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_nodes(0)
            >>> neighbors = data_graph.get_all_neighbors(nodes, 0)

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neighbor_type` is not integer.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        return self._graph_data.get_all_neighbors(node_list, neighbor_type).as_array()

    @check_gnn_get_sampled_neighbors
    def get_sampled_neighbors(self, node_list, neighbor_nums, neighbor_types):
        """
        Get sampled neighbor information.

        The api supports multi-hop neighbor sampling. That is, the previous sampling result is used as the input of
        next-hop sampling. A maximum of 6-hop are allowed.

        The sampling result is tiled into a list in the format of [input node, 1-hop sampling result,
        2-hop samling result ...]

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neighbor_nums (Union[list, numpy.ndarray]): Number of neighbors sampled per hop.
            neighbor_types (Union[list, numpy.ndarray]): Neighbor type sampled per hop.

        Returns:
            numpy.ndarray: Array of nodes.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_nodes(0)
            >>> neighbors = data_graph.get_sampled_neighbors(nodes, [2, 2], [0, 0])

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neighbor_nums` is not list or ndarray.
            TypeError: If `neighbor_types` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        return self._graph_data.get_sampled_neighbors(
            node_list, neighbor_nums, neighbor_types).as_array()

    @check_gnn_get_neg_sampled_neighbors
    def get_neg_sampled_neighbors(self, node_list, neg_neighbor_num, neg_neighbor_type):
        """
        Get `neg_neighbor_type` negative sampled neighbors of the nodes in `node_list`.

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neg_neighbor_num (int): Number of neighbors sampled.
            neg_neighbor_type (int): Specify the type of negative neighbor.

        Returns:
            numpy.ndarray: Array of nodes.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_nodes(0)
            >>> neg_neighbors = data_graph.get_neg_sampled_neighbors(nodes, 5, 0)

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neg_neighbor_num` is not integer.
            TypeError: If `neg_neighbor_type` is not integer.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        return self._graph_data.get_neg_sampled_neighbors(
            node_list, neg_neighbor_num, neg_neighbor_type).as_array()

    @check_gnn_get_node_feature
    def get_node_feature(self, node_list, feature_types):
        """
        Get `feature_types` feature of the nodes in `node_list`.

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            feature_types (Union[list, numpy.ndarray]): The given list of feature types.

        Returns:
            numpy.ndarray: array of features.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_nodes(0)
            >>> features = data_graph.get_node_feature(nodes, [1])

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `feature_types` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        if isinstance(node_list, list):
            node_list = np.array(node_list, dtype=np.int32)
        return [
            t.as_array() for t in self._graph_data.get_node_feature(
                Tensor(node_list),
                feature_types)]

    @check_gnn_get_edge_feature
    def get_edge_feature(self, edge_list, feature_types):
        """
        Get `feature_types` feature of the edges in `edge_list`.

        Args:
            edge_list (Union[list, numpy.ndarray]): The given list of edges.
            feature_types (Union[list, numpy.ndarray]): The given list of feature types.

        Returns:
            numpy.ndarray: array of features.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> edges = data_graph.get_all_edges(0)
            >>> features = data_graph.get_edge_feature(edges, [1])

        Raises:
            TypeError: If `edge_list` is not list or ndarray.
            TypeError: If `feature_types` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        if isinstance(edge_list, list):
            edge_list = np.array(edge_list, dtype=np.int32)
        return [
            t.as_array() for t in self._graph_data.get_edge_feature(
                Tensor(edge_list),
                feature_types)]

    def graph_info(self):
        """
        Get the meta information of the graph, including the number of nodes, the type of nodes,
        the feature information of nodes, the number of edges, the type of edges, and the feature information of edges.

        Returns:
            dict: Meta information of the graph. The key is node_type, edge_type, node_num, edge_num,
            node_feature_type and edge_feature_type.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        return self._graph_data.graph_info()

    @check_gnn_random_walk
    def random_walk(
            self,
            target_nodes,
            meta_path,
            step_home_param=1.0,
            step_away_param=1.0,
            default_node=-1):
        """
        Random walk in nodes.

        Args:
            target_nodes (list[int]): Start node list in random walk
            meta_path (list[int]): node type for each walk step
            step_home_param (float, optional): return hyper parameter in node2vec algorithm (Default = 1.0).
            step_away_param (float, optional): inout hyper parameter in node2vec algorithm (Default = 1.0).
            default_node (int, optional): default node if no more neighbors found (Default = -1).
                A default value of -1 indicates that no node is given.

        Returns:
            numpy.ndarray: Array of nodes.

        Examples:
            >>> import mindspore.dataset as ds
            >>>
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.random_walk([1,2], [1,2,1,2,1])

        Raises:
            TypeError: If `target_nodes` is not list or ndarray.
            TypeError: If `meta_path` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server")
        return self._graph_data.random_walk(target_nodes, meta_path, step_home_param, step_away_param,
                                            default_node).as_array()
