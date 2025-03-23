import React, { useState } from "react";
import axios from "axios";

function App() {
  const [formData, setFormData] = useState({
    Age: "",
    BMI: "",
    Cycle: "", // Irregular cycle
    Weight_Gain: "",
    Hair_Growth: "",
    Skin_Darkening: "",
    Hair_Loss: "",
    Pimples: "",
    TSH: "",
    Follicle_No_L: "",
    Follicle_No_R: "",
  });

  const [prediction, setPrediction] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict_pcos", formData);
      setPrediction(res.data.PCOS_Prediction);
    } catch (error) {
      console.error("Error:", error);
      setPrediction("Error occurred while predicting.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-2xl font-bold mb-4">PCOS Detection System</h1>
      
      <div className="grid grid-cols-2 gap-4">
        {Object.keys(formData).map((key) => (
          <input
            key={key}
            type="number"
            name={key}
            placeholder={key.replace("_", " ")}
            value={formData[key]}
            onChange={handleChange}
            className="border p-2 rounded"
          />
        ))}
      </div>

      <button onClick={handleSubmit} className="mt-4 bg-blue-500 text-white px-4 py-2 rounded">
        Predict PCOS
      </button>

      {prediction && (
        <div className="mt-4 p-4 bg-white shadow rounded">
          <h2 className="font-semibold">Prediction:</h2>
          <p>{prediction}</p>
        </div>
      )}
    </div>
  );
}

export default App;
