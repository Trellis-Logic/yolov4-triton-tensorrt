/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

 // See https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/post_processor/nvdsinfer_custombboxparser_tao.cpp#L77-L78

static bool NvDsInferParseCustomNMSTLT(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{

    float* out_nms = (float *) outputLayersInfo[0].buffer;
    const int out_class_size = detectionParams.numClassesConfigured;
    const float threshold = detectionParams.perClassThreshold[0];
    // 7 colums per row in the format [x, y, w, h, box_confidence, class_id, class_prob]
    // See example at https://github.com/isarsoft/yolov4-triton-tensorrt/blob/543fde846b2751d6ab394339e005e2754de22972/clients/python/processing.py#L76
    const int num_rows = outputLayersInfo[0].inferDims.numElements / 7;

#if 0
    std::cout << "Parser output layers" << std::endl;

    for (auto it = outputLayersInfo.begin();
          it != outputLayersInfo.end();
          it++ ) {
        const NvDsInferLayerInfo &info = *it;
        std::cout << "Name " << info.layerName << " isInput " <<
            info.isInput << " bindingIndex " << info.bindingIndex <<
            " numDims " << info.inferDims.numDims <<
            " num Elements" << info.inferDims.numElements << std::endl;

        for (unsigned int dim = 0; dim < info.inferDims.numDims; dim++ ) {
            std::cout << "    Dim " << dim << " size " << info.inferDims.d[dim] << std::endl;
        }
    }
    std::cout << "End of output layers" << std::endl;
#endif

    for (int i = 0; i < num_rows; i++) {
        float *det = out_nms + i * 7;
        float box_confidence = det[4];
        float class_confidence = det[6];
        float confidence = box_confidence * class_confidence;
        int class_id = det[5];
        float x = det[0];
        float y = det[1];
        float w = det[2];
        float h = det[3];

        // Output format for each detection is stored in the below order
        if ( confidence < threshold ) continue;
        std::cout << "adding box with confidence " << confidence << " box,class " << box_confidence
          << "," << class_confidence << " greater than threshold " << threshold << std::endl;

        std::cout << "x " << x << " y " << y << " w " << w << " h " << h << " class_id " << class_id << " box,class " << box_confidence
          << class_confidence << " greater than threshold " << threshold << std::endl;

        assert( class_id < out_class_size );
        NvDsInferObjectDetectionInfo object;
            object.classId = class_id;
            object.detectionConfidence = confidence;

            /* Clip object box co-ordinates to network resolution */
            object.left = CLIP(x * networkInfo.width, 0, networkInfo.width - 1);
            object.top = CLIP(y * networkInfo.height, 0, networkInfo.height - 1);
            object.width = CLIP(w * networkInfo.width, 0, networkInfo.width - 1);
            object.height = CLIP(h * networkInfo.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
    }

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseCustomNMSTLT (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
