import React, { useRef, useState } from "react";
import Webcam from "react-webcam";

const App = () => {
  const webcamRef = useRef(null);
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImage(imageSrc);
    sendImageToBackend(imageSrc);
  };

  const sendImageToBackend = async (imageSrc) => {
    try {
      const response = await fetch("http://localhost:8080/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageSrc }),
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error("Error sending image: ", error);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 p-6">
      <h1 className="text-2xl font-bold">Plant Disease Identification</h1>
      <Webcam ref={webcamRef} screenshotFormat="image/jpeg" className="rounded shadow-lg" />
      <button className="p-2 bg-blue-500 text-white rounded" onClick={capture}>Capture Image</button>
      {image && <img src={image} alt="Captured" className="w-64 rounded shadow-lg" />}
      {prediction && (
        <div className="text-lg font-semibold text-green-700">
          Predicted: {prediction.label} <br /> Confidence: {(prediction.confidence * 100).toFixed(2)}%
        </div>
      )}
    </div>
  );
};

export default App;
