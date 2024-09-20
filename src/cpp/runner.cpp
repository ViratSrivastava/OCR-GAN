#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <json/json.h>
#include <fstream>
#include <vector>
#include <string>

// Function to load the TensorFlow model
TF_Graph* loadModel(const char* modelPath) {
    // Implementation to load the TensorFlow model
    // Return the loaded graph
}

// Function to process an image and get predictions
std::vector<float> processImage(TF_Graph* graph, const cv::Mat& image) {
    // Implementation to preprocess the image and run inference
    // Return the output probabilities
}

// Function to convert class index to character
char indexToChar(int index) {
    if (index < 10) return '0' + index;
    if (index < 36) return 'a' + (index - 10);
    return 'A' + (index - 36);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    // Load the image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image" << std::endl;
        return 1;
    }

    // Load the TensorFlow model
    TF_Graph* graph = loadModel("path_to_saved_model");

    // Process the image
    std::vector<float> predictions = processImage(graph, image);

    // Find the class with the highest probability
    int predictedClass = std::max_element(predictions.begin(), predictions.end()) - predictions.begin();
    char predictedChar = indexToChar(predictedClass);

    // Create JSON output
    Json::Value root;
    root["predicted_class"] = predictedClass;
    root["predicted_char"] = std::string(1, predictedChar);
    root["confidence"] = predictions[predictedClass];

    // Write JSON to file
    std::ofstream jsonFile("output.json");
    jsonFile << root;
    jsonFile.close();

    // Write to XML file
    std::ofstream xmlFile("output.xml");
    xmlFile << "<result>\n";
    xmlFile << "  <predicted_class>" << predictedClass << "</predicted_class>\n";
    xmlFile << "  <predicted_char>" << predictedChar << "</predicted_char>\n";
    xmlFile << "  <confidence>" << predictions[predictedClass] << "</confidence>\n";
    xmlFile << "</result>\n";
    xmlFile.close();

    // Write to text file
    std::ofstream txtFile("output.txt");
    txtFile << "Predicted class: " << predictedClass << "\n";
    txtFile << "Predicted character: " << predictedChar << "\n";
    txtFile << "Confidence: " << predictions[predictedClass] << "\n";
    txtFile.close();

    std::cout << "Processing complete. Results saved to output files." << std::endl;

    return 0;
}