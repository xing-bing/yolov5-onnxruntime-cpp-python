#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <vector>
#include <string>
using namespace std;
using namespace cv;
using namespace Ort;

//自定义配置结构
struct Configuration{

    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    string modelpath;

};

//定义BoxInfo 结构类型
typedef struct BoxInfo{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class YOLOv5{
public:
    YOLOv5(Configuration config);
    void detect(Mat &frame);
private:
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    int inpWidth;
    int inpHeight;
    int nout;
    int num_proposal;
    int num_classes;
    string classes[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
							"train", "truck", "boat", "traffic light", "fire hydrant",
							"stop sign", "parking meter", "bench", "bird", "cat", "dog",
							"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
							"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
							"skis", "snowboard", "sports ball", "kite", "baseball bat",
							"baseball glove", "skateboard", "surfboard", "tennis racket",
							"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
							"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
							"hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
							"bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
							"remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
							"sink", "refrigerator", "book", "clock", "vase", "scissors",
							"teddy bear", "hair drier", "toothbrush"};
    const bool keep_ratio = true;
    vector<float> input_image_;
    void normalize_(Mat img);
    void nms(vector<BoxInfo> &input_boxes);
    Mat resize_image(Mat srcing, int *newh, int *neww, int *top, int *left);
    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1"); //初始化环境
    Session *ort_session = nullptr;
    SessionOptions sessionOptions = SessionOptions();
    vector<char *> input_names;
    vector<char *> output_names;
    vector<vector<int64_t>> input_node_dims;
    vector<vector<int64_t>> output_node_dims;

};

YOLOv5::YOLOv5(Configuration config){
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->objThreshold = config.objThreshold;
    this->num_classes = sizeof(this->classes) / sizeof(this->classes[0]);
    this->inpHeight = 640;
    this->inpWidth = 640;
    string model_path = config.modelpath;

    //开启cuda加速
    OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    ort_session = new Session(env, (const char *)model_path.c_str(), sessionOptions);
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    //输入输出节点分配内存
    AllocatorWithDefaultOptions allocator;
    for(int i=0;i<numInputNodes;i++){
        input_names.push_back(ort_session->GetInputName(i, allocator));
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }   
    for(int i=0;i<numOutputNodes;i++){
        output_names.push_back(ort_session->GetOutputName(i, allocator));
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);

    }
    this->inpHeight = input_node_dims[0][2];
    this->inpWidth = input_node_dims[0][3];
    this->nout = output_node_dims[0][2];
    this->num_proposal = output_node_dims[0][1];
}

Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left){
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight;
    *neww = this->inpWidth;
    Mat dstimg;
    if (this->keep_ratio && srch != srcw){
         float hw_scale = (float) srch / srcw;
         if (hw_scale > 1){
            *newh = this->inpHeight;
            *neww = int(this->inpWidth / hw_scale);
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *left = int((this->inpWidth - *neww) * 0.5);
            copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);

         }
         else{
            *newh = (int)this->inpHeight * hw_scale;
            *neww = this->inpWidth;
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *top = (int)(this->inpHeight - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);

         }
    }
    else{
        resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
    }
    return dstimg;
}

void YOLOv5::normalize_(Mat img){
    int row = img.rows;
    int col = img.cols;

    this->input_image_.resize(row * col * img.channels());
    for(int c=0;c<3;c++){
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
                this->input_image_[c * row * col + i * col + j] = pix / 255.0;
            }
        }
    }
}

void YOLOv5::nms(vector<BoxInfo> &input_boxes){
    //对所有的预测框按照置信度从大到小排序
    sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b){return a.score > b.score;});
    vector<float> vArea(input_boxes.size());
    for (int i=0;i<input_boxes.size(); i++){
        vArea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1) * (input_boxes[i].y2 - input_boxes[i].y1 + 1);
    }
    vector<bool> isSuppressed(input_boxes.size(), false);
    for(int i=0;i<input_boxes.size(); i++){
        if (isSuppressed[i]){continue;};
        for(int j=i+1;j<input_boxes.size();j++){
            if (isSuppressed[j]){continue;}
            float xx1 = max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = min(input_boxes[i].y2, input_boxes[j].y2);

            float w = max(0.0f, xx2 - xx1 + 1);
            float h = max(0.0f, yy2 - yy1 + 1);
            float inter = w * h;

            if (input_boxes[i].label == input_boxes[j].label){
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr>=this->nmsThreshold){
                    isSuppressed[j] = true;
                }
            }
        }
    }

    int idx_t = 0;
    input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo &f){return isSuppressed[idx_t++];}), input_boxes.end());
}

void YOLOv5::detect(Mat &frame){
    int newh = 0, neww = 0, padh = 0, padw = 0;
    Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
    this->normalize_(dstimg);
    array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};

    //创建输入tensor
    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

    vector<Value> ort_outputs = ort_session->Run(RunOptions{nullptr}, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
    vector<BoxInfo> generate_boxes;
    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
    float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    for(int i=0;i<num_proposal;i++){
        int index = i * nout;
        float obj_conf = pdata[index + 4];
        if (obj_conf > this->objThreshold){
            int class_idx = 0;
            float max_class_socre = 0;
            for(int k=0;k<this->num_classes;k++){
                if(pdata[k+index+5] > max_class_socre){
                    max_class_socre = pdata[k + index + 5];
                    class_idx = k;
                }
            }
            max_class_socre *= obj_conf;
            if (max_class_socre > this->confThreshold){
                float cx = pdata[index];
                float cy = pdata[index+1];
                float w = pdata[index+2];
                float h = pdata[index+3];

                float xmin = (cx - padw - 0.5 * w) * ratiow;
                float ymin = (cy - padh - 0.5 * h) * ratioh;
                float xmax = (cx - padw + 0.5 * w) * ratiow;
                float ymax = (cy - padh + 0.5 * h) * ratioh;

                generate_boxes.push_back(BoxInfo{xmin, ymin, xmax, ymax, max_class_socre, class_idx});
            }
        }
    }
    nms(generate_boxes);
    for(size_t i=0;i<generate_boxes.size();i++){
        int xmin = int(generate_boxes[i].x1);
        int ymin = int(generate_boxes[i].y1);
        int xmax = int(generate_boxes[i].x2);
        int ymax = int(generate_boxes[i].y2);
        rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
        string label = format("%.2f", generate_boxes[i].score);
        label = this->classes[generate_boxes[i].label] + ":" + label;
        putText(frame, label, Point(xmin, ymin-5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);

    }
}


int main(){
    clock_t startTime, endTime;
    Configuration yolo_nets = {0.3, 0.5, 0.3, "yolov5s.onnx"};
    YOLOv5 yolo_model(yolo_nets);
    string imgpath = "bus.jpg";
    Mat srcimg = imread(imgpath);

    yolo_model.detect(srcimg);
    imwrite("result_ort.jpg", srcimg);
    return 0;

}