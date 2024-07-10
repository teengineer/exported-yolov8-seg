import * as tf from "@tensorflow/tfjs";
import React, { useState, useEffect, useRef } from "react";

import "./style/App.css";
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detectFrame } from "./utils/detect";

const App = () => {
  const [loading, setLoading] = useState({ loadiclsng: true, progress: 0 }); // loading state
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const canvasRef = useRef(null);

  // model configs
  const modelName = "best";

  useEffect(() => {
    tf.ready().then(async () => {
      const yolov8 = await tf.loadGraphModel(
        `${window.location.href}/${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      ); // load model

      // warming up model
      const dummyInput = tf.randomUniform(yolov8.inputs[0].shape, 0, 1, "float32"); // random input
      const warmupResults = yolov8.execute(dummyInput);

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yolov8,
        inputShape: yolov8.inputs[0].shape,
        outputShape: warmupResults.map((e) => e.shape),
      }); // set model & input shape

      tf.dispose([warmupResults, dummyInput]); // cleanup memory
    });
  }, []);

  return (
    <div className="App">
      {loading.loading && <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>}
      <div className="header">
        <h1>📷 YOLOv8 Live Roof Segmentation App</h1>
        <p>
          YOLOv8 live detection and segmentation application on browser powered by <code>tensorflow.js</code>
        </p>
        <p>
          Trained Model : <code className="code">{modelName}</code>
        </p>
      </div>
      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad={() => detectFrame(imageRef.current, model, canvasRef.current)}
        />
        <canvas width={model.inputShape[2]} height={model.inputShape[1]} ref={canvasRef} />
      </div>

      <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} />
    </div>
  );
};

export default App;
