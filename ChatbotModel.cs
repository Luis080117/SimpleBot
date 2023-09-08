using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Chatbot
{
    public class ChatbotModel
    {
        private MLContext _context;
        private ITransformer _model;

        public ChatbotModel()
        {
            _context = new MLContext();
        }

        public void TrainModel(string datasetPath)
        {
            var data = _context.Data.LoadFromTextFile<ChatData>(datasetPath, separatorChar: ',');

            var pipeline = _context.Transforms.Concatenate("Features", "Question", "Intent")
                .Append(_context.Text.TextCategorizer(labelColumnName: "Intent", maxSentenceLength: 100));

            _model = pipeline.Fit(data);

            _context.Model.Save(_model, data.Schema, "chatbot_model.zip");

            Console.WriteLine("Model trained and saved successfully.");
        }

        public string Respond(string question)
        {
            var sizeBedroomData = new ChatData
            {
                Question = question
            };

            var predictionFunction = _context.Model.CreatePredictionEngine<ChatData, ChatPrediction>(_model);

            var predictedIntent = predictionFunction.Predict(sizeBedroomData);

            return predictedIntent.Intent;
        }
    }

    public class ChatData
    {
        public string Question { get; set; }
        public string Intent { get; set; }
    }

    public class ChatPrediction
    {
        [ColumnName("Score")]
        public string Intent;
    }
}
