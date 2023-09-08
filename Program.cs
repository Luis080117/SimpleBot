var chatbotModel = new ChatbotModel();
chatbotModel.TrainModel("chatbot_dataset.csv");

var question = "What is the monthly payment for a \$300,000 mortgage?";
var intent = chatbotModel.Respond(question);

Console.WriteLine(intent);
