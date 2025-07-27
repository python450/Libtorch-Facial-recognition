#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
namespace fs = std::filesystem;
class ImageDataset : public torch::data::Dataset<ImageDataset>
{
private:
	std::vector<std::string> class_names_;
	std::vector<std::string> _images;
	std::vector<int> _labels;
public:
	explicit ImageDataset(std::string in_path)
	{
		for (const auto& class_dir : fs::directory_iterator(in_path))
		{
			if (!fs::is_directory(class_dir)) continue;
			auto label = class_names_.size();
			class_names_.push_back(class_dir.path().filename().string());
			for (const auto& img_path : fs::directory_iterator(class_dir.path()))
			{
				if (img_path.path().extension() != ".png" && img_path.path().extension() != ".jpg") continue;
				_images.push_back(img_path.path().string());
				_labels.push_back(label);
			}
			if (class_names_.size() >= 2) break;
		}
	}
	torch::data::Example<> get(size_t index) override
	{
		auto mat = cv::imread(_images[index], cv::IMREAD_COLOR);
		cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
		cv::resize(mat, mat, cv::Size(28, 28));
		mat.convertTo(mat, CV_32FC3, 1.0 / 255.0);
		auto mat_tensor = torch::from_blob(mat.data, { mat.rows, mat.cols, 3 }, torch::kFloat32).permute({ 2, 0, 1 });
		mat_tensor = torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 })(mat_tensor);
		auto label_tensor = torch::full({ 1 }, _labels[index], torch::kInt64);
		return { mat_tensor, label_tensor };
	}
	torch::optional<size_t> size() const override
	{
		return _images.size();
	}
	const std::vector<std::string>& class_names() const
	{
		return class_names_;
	}
};
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
	CNN net;
	std::string de;
public:
	Box(auto in_de) : de(in_de)
	{
		torch::manual_seed(1);
		torch::set_num_threads(5);
		torch::set_num_interop_threads(5);
	}
	~Box()
	{
		std::cin.get();
		std::cin.get();
	}
	void train(auto data_load)
	{
		torch::optim::Adam op(net->parameters(), torch::optim::AdamOptions(0.001).weight_decay(1e-4));
		auto de2 = torch::kCPU;
		if (de == "cuda")
		{
			de2 = torch::kCUDA;
		}
		net->to(de2);
		std::cout << "训练开始" << std::endl;
		std::ofstream out("./models/model.log", std::ios::trunc);
		auto epoch = 0;
		while (true)
		{
			++epoch;
			net->train();
			auto running_loss = 0.0;
			auto co = 0;
			for (auto& batch : *data_load)
			{
				op.zero_grad();
				auto data = batch.data.to(de2);
				auto target = batch.target.to(de2).squeeze();
				auto output = net->forward(data);
				auto loss = torch::nll_loss(output, target);
				loss.backward();
				op.step();
				running_loss += loss.item<float>();
				co += (output.argmax(1).cpu() == target).sum().item<int>();
			}
			std::cout << "Epoch: " << epoch << "\t|\t" << "Loss: " << running_loss << std::endl;
			if (epoch % 10 == 0)
			{
				auto name = std::to_string(epoch) + "_CNN.pt";
				auto path = "./models/" + name;
				torch::save(net, path);
				std::cout << "模型 " << name << " 已保存" << std::endl;
				out << "模型 " << name << " Loss: " << running_loss << std::endl;
			}
		}
		out.close();
	}
};
int main()
{
	try
	{
		std::ios::sync_with_stdio(false);
		auto datasets = ImageDataset("./train").map(torch::data::transforms::Stack<>());
		auto data_load = torch::data::make_data_loader(std::move(datasets), torch::data::DataLoaderOptions().batch_size(128).workers(4));
		Box box("cpu");
		box.train(std::move(data_load));
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
	return 0;
}
