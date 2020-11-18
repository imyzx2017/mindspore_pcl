# Demo of Image Classification

The following describes how to use the MindSpore Lite C++ APIs (Android JNIs) and MindSpore Lite image classification models to perform on-device inference, classify the content captured by a device camera, and display the most possible classification result on the application's image preview screen.


### Running Dependencies

- Android Studio 3.2 or later (Android 4.0 or later is recommended.)
- Native development kit (NDK) 21.3
- CMake 3.10.2 [CMake](https://cmake.org/download) 
- Android software development kit (SDK) 26 or later
- JDK 1.8 or later

### Building and Running

1. Load the sample source code to Android Studio and install the corresponding SDK. (After the SDK version is specified, Android Studio automatically installs the SDK.)

    ![start_home](images/home.png)

    Start Android Studio, click `File > Settings > System Settings > Android SDK`, and select the corresponding SDK. As shown in the following figure, select an SDK and click `OK`. Android Studio automatically installs the SDK.

    ![start_sdk](images/sdk_management.png)

    (Optional) If an NDK version issue occurs during the installation, manually download the corresponding [NDK version](https://developer.android.com/ndk/downloads) (the version used in the sample code is 21.3). Specify the SDK location in `Android NDK location` of `Project Structure`.

    ![project_structure](images/project_structure.png)

2. Connect to an Android device and runs the image classification application.

    Connect to the Android device through a USB cable for debugging. Click `Run 'app'` to run the sample project on your device.

    ![run_app](images/run_app.PNG)

    For details about how to connect the Android Studio to a device for debugging, see <https://developer.android.com/studio/run/device?hl=zh-cn>.

    The mobile phone needs to be turn on "USB debugging mode" before Android Studio can recognize the mobile phone. Huawei mobile phones generally turn on "USB debugging model" in Settings > system and update > developer Options > USB debugging.

3. Continue the installation on the Android device. After the installation is complete, you can view the content captured by a camera and the inference result.

    ![result](images/app_result.jpg)

## Detailed Description of the Sample Program  

This image classification sample program on the Android device includes a Java layer and a JNI layer. At the Java layer, the Android Camera 2 API is used to enable a camera to obtain image frames and process images. At the JNI layer, the model inference process is completed in [Runtime](https://www.mindspore.cn/tutorial/lite/en/master/use/runtime.html).

### Sample Program Structure

```
app
│
├── src/main
│   ├── assets # resource files
|   |   └── mobilenetv2.ms # model file
│   |
│   ├── cpp # main logic encapsulation classes for model loading and prediction
|   |   |
|   |   ├── MindSporeNetnative.cpp # JNI methods related to MindSpore calling
│   |   └── MindSporeNetnative.h # header file
│   |
│   ├── java # application code at the Java layer
│   │   └── com.mindspore.himindsporedemo 
│   │       ├── gallery.classify # implementation related to image processing and MindSpore JNI calling
│   │       │   └── ...
│   │       └── widget # implementation related to camera enabling and drawing
│   │           └── ...
│   │   
│   ├── res # resource files related to Android
│   └── AndroidManifest.xml # Android configuration file
│
├── CMakeList.txt # CMake compilation entry file
│
├── build.gradle # Other Android configuration file
├── download.gradle # MindSpore version download
└── ...
```

### Configuring MindSpore Lite Dependencies

When MindSpore C++ APIs are called at the Android JNI layer, related library files are required. You can use MindSpore Lite [source code compilation](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html) to generate the MindSpore Lite version. In this case, you need to use the compile command of generate with image preprocessing module.

In this example, the build process automatically downloads the `mindspore-lite-1.0.1-runtime-arm64-cpu` by the `app/download.gradle` file and saves in the `app/src/main/cpp` directory.

Note: if the automatic download fails, please manually download the relevant library files and put them in the corresponding location.

mindspore-lite-1.0.1-runtime-arm64-cpu.tar.gz [Download link](https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/lite/android_aarch64/mindspore-lite-1.0.1-runtime-arm64-cpu.tar.gz)

```
android{
    defaultConfig{
        externalNativeBuild{
            cmake{
                arguments "-DANDROID_STL=c++_shared"
            }
        }

        ndk{ 
            abiFilters'armeabi-v7a', 'arm64-v8a'  
        }
    }
}
```

Create a link to the `.so` library file in the `app/CMakeLists.txt` file:

```
# ============== Set MindSpore Dependencies. =============
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/third_party/flatbuffers/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION})
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include/ir/dtype)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include/schema)

add_library(mindspore-lite SHARED IMPORTED )
add_library(minddata-lite SHARED IMPORTED )

set_target_properties(mindspore-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libmindspore-lite.so)
set_target_properties(minddata-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libminddata-lite.so)
# --------------- MindSpore Lite set End. --------------------

# Link target library.       
target_link_libraries(
    ...
     # --- mindspore ---
        minddata-lite
        mindspore-lite
    ...
)
```

### Downloading and Deploying a Model File

In this example, the  download.gradle File configuration auto download `mobilenetv2.ms `and placed in the 'app/libs/arm64-v8a' directory.

Note: if the automatic download fails, please manually download the relevant library files and put them in the corresponding location.

mobilenetv2.ms [mobilenetv2.ms]( https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.ms)

### Compiling On-Device Inference Code

Call MindSpore Lite C++ APIs at the JNI layer to implement on-device inference.

The inference code process is as follows. For details about the complete code, see `src/cpp/MindSporeNetnative.cpp`. 

1. Load the MindSpore Lite model file and build the context, session, and computational graph for inference.  

   - Load a model file. Create and configure the context for model inference.

     ```cpp
     // Buffer is the model data passed in by the Java layer
     jlong bufferLen = env->GetDirectBufferCapacity(buffer);
     char *modelBuffer = CreateLocalModelBuffer(env, buffer);  
     ```

   - Create a session.

     ```cpp
     void **labelEnv = new void *;
     MSNetWork *labelNet = new MSNetWork;
     *labelEnv = labelNet;
     
     // Create context.
     mindspore::lite::Context *context = new mindspore::lite::Context;
     context->thread_num_ = num_thread;
     
     // Create the mindspore session.
     labelNet->CreateSessionMS(modelBuffer, bufferLen, "device label", context);
     delete(context);
     
     ```

   - Load the model file and build a computational graph for inference.

     ```cpp
     void MSNetWork::CreateSessionMS(char* modelBuffer, size_t bufferLen, std::string name, mindspore::lite::Context* ctx)
     {
         CreateSession(modelBuffer, bufferLen, ctx);  
         session = mindspore::session::LiteSession::CreateSession(ctx);
         auto model = mindspore::lite::Model::Import(modelBuffer, bufferLen);
         int ret = session->CompileGraph(model);
     }
     ```

2. Convert the input image into the Tensor format of the MindSpore model. 

   Convert the image data to be detected into the Tensor format of the MindSpore model.

     ```cpp
    if (!BitmapToLiteMat(env, srcBitmap, &lite_mat_bgr)) {
     MS_PRINT("BitmapToLiteMat error");
     return NULL;
    }
    if (!PreProcessImageData(lite_mat_bgr, &lite_norm_mat_cut)) {
     MS_PRINT("PreProcessImageData error");
     return NULL;
    }

    ImgDims inputDims;
    inputDims.channel = lite_norm_mat_cut.channel_;
    inputDims.width = lite_norm_mat_cut.width_;
    inputDims.height = lite_norm_mat_cut.height_;

    // Get the mindsore inference environment which created in loadModel().
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    if (labelEnv == nullptr) {
     MS_PRINT("MindSpore error, labelEnv is a nullptr.");
     return NULL;
    }
    MSNetWork *labelNet = static_cast<MSNetWork *>(*labelEnv);

    auto mSession = labelNet->session();
    if (mSession == nullptr) {
     MS_PRINT("MindSpore error, Session is a nullptr.");
     return NULL;
    }
    MS_PRINT("MindSpore get session.");

    auto msInputs = mSession->GetInputs();
    if (msInputs.size() == 0) {
     MS_PRINT("MindSpore error, msInputs.size() equals 0.");
     return NULL;
    }
    auto inTensor = msInputs.front();

    float *dataHWC = reinterpret_cast<float *>(lite_norm_mat_cut.data_ptr_);
    // Copy dataHWC to the model input tensor.
    memcpy(inTensor->MutableData(), dataHWC,
         inputDims.channel * inputDims.width * inputDims.height * sizeof(float));
    ```

3. Perform inference on the input tensor based on the model, obtain the output tensor, and perform post-processing.    

   - Perform graph execution and on-device inference.

     ```cpp
     // After the model and image tensor data is loaded, run inference.
     auto status = mSession->RunGraph();
     ```

   - Obtain the output data.

     ```cpp
     auto names = mSession->GetOutputTensorNames();
     std::unordered_map<std::string,mindspore::tensor::MSTensor *> msOutputs;
     for (const auto &name : names) {
         auto temp_dat =mSession->GetOutputByTensorName(name);
         msOutputs.insert(std::pair<std::string, mindspore::tensor::MSTensor *> {name, temp_dat});
       }
     std::string retStr = ProcessRunnetResult(msOutputs, ret);
     ```

   - Perform post-processing of the output data.

     ```cpp
     std::string ProcessRunnetResult(const int RET_CATEGORY_SUM, const char *const labels_name_map[],
              std::unordered_map<std::string, mindspore::tensor::MSTensor *> msOutputs) {
      // Get the branch of the model output.
      // Use iterators to get map elements.
      std::unordered_map<std::string, mindspore::tensor::MSTensor *>::iterator iter;
      iter = msOutputs.begin();

      // The mobilenetv2.ms model output just one branch.
      auto outputTensor = iter->second;

      int tensorNum = outputTensor->ElementsNum();
      MS_PRINT("Number of tensor elements:%d", tensorNum);

      // Get a pointer to the first score.
      float *temp_scores = static_cast<float *>(outputTensor->MutableData());
      float scores[RET_CATEGORY_SUM];
      for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
       scores[i] = temp_scores[i];
      }

      float unifiedThre = 0.5;
      float probMax = 1.0;
      for (size_t i = 0; i < RET_CATEGORY_SUM; ++i) {
       float threshold = g_thres_map[i];
       float tmpProb = scores[i];
       if (tmpProb < threshold) {
        tmpProb = tmpProb / threshold * unifiedThre;
       } else {
        tmpProb = (tmpProb - threshold) / (probMax - threshold) * unifiedThre + unifiedThre;
      }
       scores[i] = tmpProb;
     }

      for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
      if (scores[i] > 0.5) {
          MS_PRINT("MindSpore scores[%d] : [%f]", i, scores[i]);
       }
      }

      // Score for each category.
      // Converted to text information that needs to be displayed in the APP.
      std::string categoryScore = "";
      for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
       categoryScore += labels_name_map[i];
       categoryScore += ":";
       std::string score_str = std::to_string(scores[i]);
       categoryScore += score_str;
       categoryScore += ";";
      }
        return categoryScore;
     }
     
     ```
