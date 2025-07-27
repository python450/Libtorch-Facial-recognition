#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
class CNNImpl : public torch::nn::Module
{
private:
	torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
	torch::nn::Dropout dropout{ nullptr };
public:
	CNNImpl()
	{
		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
		conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
		fc1 = register_module("fc1", torch::nn::Linear(128 * 3 * 3, 128));
		dropout = register_module("dropout", torch::nn::Dropout(0.5));
		fc2 = register_module("fc2", torch::nn::Linear(128, 2));
	}
	auto forward(auto x)
	{
		x = torch::silu(conv1(x));
		x = torch::max_pool2d(x, 2);
		x = torch::silu(conv2(x));
		x = torch::max_pool2d(x, 2);
		x = torch::silu(conv3(x));
		x = torch::max_pool2d(x, 2);
		x = x.view({ -1, 128 * 3 * 3 });
		x = torch::silu(fc1(x));
		x = dropout(x);
		x = fc2(x);
		return torch::log_softmax(x, 1);
	}
};
TORCH_MODULE(CNN);
class Box
{
private:
	std::string file;
	std::vector<std::string> list;
protected:
	auto transform(auto img)
	{
		cv::resize(img, img, cv::Size(28, 28));
		auto tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
		tensor = tensor.permute({ 2, 0, 1 }).to(torch::kFloat32).div_(255);
		tensor = (tensor - 0.5) / 0.5;
		return tensor;
	}
public:
	Box()
	{
		list.push_back("Female");
		list.push_back("Male");
	}
	~Box() = default;
	static auto com(const auto& a, const auto& b)
	{
		return a.area() > b.area();
	}
	auto saveImage(auto face, auto image)
	{
		face = face & cv::Rect(0, 0, image.cols, image.rows);
		if (face.width <= 0 || face.height <= 0) return false;
		auto img = image(face);
		file = "output.jpg";
		return imwrite(file, img);
	}
	auto predict()
	{
		CNN model_cnn;
		torch::load(model_cnn, "./models/420_CNN.pt");
		model_cnn->eval();
		auto img = cv::imread(file);
		auto tensor = transform(img).unsqueeze(0);
		torch::NoGradGuard no_grad;
		auto output = model_cnn->forward(tensor);
		auto max_result = output.max(1);
		auto pred_result = std::get<1>(max_result).item<int>();
		auto prob = torch::softmax(output, 1)[0][pred_result].item<double>() * 100;
		return list[pred_result] + ":" + std::to_string((int)prob) + "%";
	}
};
int main()
{
	try
	{
		std::ios::sync_with_stdio(false);
		auto face_c = cv::CascadeClassifier{};
		if (!face_c.load("haarcascade_frontalface_alt2.xml"))
		{
			std::cerr << "模型加载失败！" << std::endl;
			std::cin.get();
			std::cin.get();
			return 11;
		}
		auto eye_c = cv::CascadeClassifier{};
		if (!eye_c.load("haarcascade_eye_tree_eyeglasses.xml"))
		{
			std::cerr << "模型加载失败！" << std::endl;
			std::cin.get();
			std::cin.get();
			return 11;
		}
		cv::VideoCapture cap(0);
		if (!cap.isOpened())
		{
			std::cerr << "无法打开摄像头！" << std::endl;
			std::cin.get();
			std::cin.get();
			return 11;
		}
		cap.set(cv::CAP_PROP_FRAME_WIDTH, 1500);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1200);
		cap.set(cv::CAP_PROP_FPS, 60);
		auto frame = cv::Mat();
		Box box;
		while (true)
		{
			cap >> frame;
			if (frame.empty()) break;
			auto grey = cv::Mat{};
			cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
			cv::equalizeHist(grey, grey);
			auto faces = std::vector<cv::Rect>{};
			auto eyes = std::vector<cv::Rect>{};
			face_c.detectMultiScale(grey, faces, 1.1, 5, 0, cv::Size(30, 30));
			if (faces.empty())
			{
				eye_c.detectMultiScale(grey, eyes, 1.1, 5, 0, cv::Size(20, 20));
				if (eyes.size() >= 2)
				{
					sort(eyes.begin(), eyes.end(), Box::com);
					auto eye1 = eyes[0];
					auto eye2 = eyes[1];
					auto eye1_center = cv::Point(eye1.x + eye1.width / 2, eye1.y + eye1.height / 2);
					auto eye2_center = cv::Point(eye2.x + eye2.width / 2, eye2.y + eye2.height / 2);
					auto face_width = static_cast<int>(cv::norm(eye1_center - eye2_center) * 2.2);
					auto face_height = face_width * 1.2;
					auto face_center = (eye1_center + eye2_center) * 0.5;
					auto e_face = cv::Rect(face_center.x - face_width / 2, face_center.y - face_height / 3, face_width, face_height);
					rectangle(frame, e_face, cv::Scalar(0, 255, 0), 2);
					box.saveImage(e_face, frame);
					auto text = box.predict();
					putText(frame, text, cv::Point(e_face.x, e_face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				}
			}
			else
			{
				for (const auto& face : faces)
				{
					rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
					box.saveImage(face, frame);
					auto text = box.predict();
					putText(frame, text, cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				}
			}
			imshow("人脸识别", frame);
			if (cv::waitKey(1) == 'q') break;
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
	return 0;
}
