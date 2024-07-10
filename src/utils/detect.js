import * as tf from "@tensorflow/tfjs";
import { renderBoxes, Colors } from "./renderBox";
import labels from "./labels.json";

const numClass = labels.length;
const colors = new Colors();

/**
 * Preprocess image / frame before forwarded into the model
 * @param {HTMLVideoElement|HTMLImageElement} source
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @returns input tensor, xRatio and yRatio
 */
const preprocess = (source, modelWidth, modelHeight) => {
  let xRatio, yRatio; // ratios for boxes

  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    // padding image to square => [n, m] to [n, n], n > m
    const [h, w] = img.shape.slice(0, 2); // get source width and height
    const maxSize = Math.max(w, h); // get max size
    const imgPadded = img.pad([
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    xRatio = maxSize / w; // update xRatio
    yRatio = maxSize / h; // update yRatio

    return tf.image
      .resizeBilinear(imgPadded, [ modelHeight, modelWidth]) // resize frame
      .div(255.0) // normalize
      .expandDims(0); // add batch
  });

  return [input, xRatio, yRatio];
};

/**
 * Function to detect image.
 * @param {HTMLImageElement} source Source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 * @param {VoidFunction} callback Callback function to run after detect frame is done
 */
export const detectFrame = async (source, model, canvasRef, callback = () => {}) => {
  const [modelHeight, modelWidth] = model.inputShape.slice(1, 3); // get model width and height
  const [modelSegHeight, modelSegWidth, modelSegChannel] = model.outputShape[0].slice(1);

  tf.engine().startScope(); // start scoping tf engine

  const [input, xRatio, yRatio] = preprocess(source, modelWidth, modelHeight); // do preprocessing

  const res = model.net.execute(input); // execute model
  const transRes = tf.tidy(() => res[1].transpose([0, 2, 1]).squeeze()); // transpose main result
  const transSegMask = tf.tidy(() => res[0].transpose([0, 3, 1, 2]).squeeze()); // transpose segmentation mask result

  const boxes = tf.tidy(() => {
    const w = transRes.slice([0, 2], [-1, 1]);
    const h = transRes.slice([0, 3], [-1, 1]);
    const x1 = tf.sub(transRes.slice([0, 0], [-1, 1]), tf.div(w, 2)); //x1
    const y1 = tf.sub(transRes.slice([0, 1], [-1, 1]), tf.div(h, 2)); //y1
    return tf
      .concat(
        [
          y1,
          x1,
          tf.add(y1, h), //y2
          tf.add(x1, w), //x2
        ],
        1
      ) // [y1, x1, y2, x2]
      .squeeze(); // [n, 4]
  }); // get boxes [y1, x1, y2, x2]
  const [scores, classes] = tf.tidy(() => {
    const rawScores = transRes.slice([0, 4], [-1, numClass]); // [n, numClass]
    const maxScores = rawScores.max(-1); // max along the last axis
    const argMaxClasses = rawScores.argMax(-1); // argmax along the last axis

    return [maxScores, argMaxClasses];
  }); 
  const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // do nms to filter boxes
  const detReady = tf.tidy(() =>
    tf.concat(
      [
        boxes.gather(nms, 0),
        scores.gather(nms, 0).expandDims(1),
        classes.gather(nms, 0).expandDims(1),
      ],
      1 // axis
    )
  ); // indexing selected boxes, scores and classes from NMS result
  const masks = tf.tidy(() => {
    const sliced = transRes.slice([0, 4 + numClass], [-1, modelSegChannel]).squeeze(); // slice mask from every detection [m, mask_size]
    return sliced
      .gather(nms, 0) // get selected mask from NMS result
      .matMul(transSegMask.reshape([modelSegChannel, -1])) // matmul mask with segmentation mask result [n, mask_size] x [mask_size, h x w] => [n, h x w]
      .reshape([nms.shape[0], modelSegHeight, modelSegWidth]); // reshape back [n, h x w] => [n, h, w]
  }); // processing mask

  let overlay = tf.zeros([modelHeight, modelWidth, 4]); // initialize overlay to draw mask

  for (let i = 0; i < detReady.shape[0]; i++) {
    const rowData = detReady.slice([i, 0], [1, 6]); // get every first 6 element from every row
    let [y1, x1, y2, x2, label] = rowData.dataSync(); // [y1, x1, y2, x2, score, label]
    const color = colors.get(label); // get label color
    const downSampleBox = [
      Math.floor((y1 * modelSegHeight) / modelHeight), // y
      Math.floor((x1 * modelSegWidth) / modelWidth), // x
      Math.round(((y2 - y1) * modelSegHeight) / modelHeight), // h
      Math.round(((x2 - x1) * modelSegWidth) / modelWidth), // w
    ]; // downsampled box (box ratio at model output)
    const upSampleBox = [
      Math.floor(y1 * yRatio), // y
      Math.floor(x1 * xRatio), // x
      Math.round((y2 - y1) * yRatio), // h
      Math.round((x2 - x1) * xRatio), // w
    ]; // upsampled box (box ratio to draw)

    const proto = tf.tidy(() => {
      const sliced = masks.slice(
        [
          i,
          downSampleBox[0] >= 0 ? downSampleBox[0] : 0,
          downSampleBox[1] >= 0 ? downSampleBox[1] : 0,
        ],
        [
          1,
          downSampleBox[0] + downSampleBox[2] <= modelSegHeight
            ? downSampleBox[2]
            : modelSegHeight - downSampleBox[0],
          downSampleBox[1] + downSampleBox[3] <= modelSegWidth
            ? downSampleBox[3]
            : modelSegWidth - downSampleBox[1],
        ]
      ); // coordinate to slice mask from proto
      return sliced.squeeze().expandDims(-1); // sliced proto [h, w, 1]
    });
    const upsampleProto = tf.image.resizeBilinear(proto, [upSampleBox[2], upSampleBox[3]]); // resizing proto to drawing size
    const mask = tf.tidy(() => {
      const padded = upsampleProto.pad([
        [upSampleBox[0], modelHeight - (upSampleBox[0] + upSampleBox[2])],
        [upSampleBox[1], modelWidth - (upSampleBox[1] + upSampleBox[3])],
        [0, 0],
      ]); // padding proto to canvas size
      return padded.less(0.5); // make boolean mask from proto to indexing overlay
    }); // final boolean mask

    overlay = tf.tidy(() => {
      const newOverlay = overlay.where(mask, [...Colors.hexToRgba(color), 150]); // indexing overlay from mask with RGBA code
      overlay.dispose(); // dispose old overlay tensor (free memory)
      return newOverlay; // return new overlay
    }); // new overlay
    tf.dispose([rowData, proto, upsampleProto, mask]); // dispose unused tensor to free memory
  }

  const maskImg = new ImageData(
    new Uint8ClampedArray(await overlay.data()), // tensor to array
    modelWidth, 
    modelHeight
  ); // create image data from mask overlay

  const ctx = canvasRef.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
  ctx.putImageData(maskImg, 0, 0); // render overlay to canvas

  //#region EDGE DETECTION
  // Apply Canny edge detection
  const src = cv.imread(canvasRef);
  const dst = new cv.Mat();
  cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(src, src, { width: 5, height: 5 }, 2, 2);
  cv.Canny(src, dst, 50, 150, 3, false);
  cv.imshow(canvasRef, dst);

  // Extract contours
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(dst, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  // Create an array to store edge coordinates
  const edgeCoordinates = [];

  //Iterate through contours and collect edge coordinates
  for (let i = 0; i < contours.size(); ++i) {
    const contour = contours.get(i);
    const contourCoordinates = [];
    for (let j = 0; j < contour.data32S.length; j += 2) {
      const x = contour.data32S[j];
      const y = contour.data32S[j + 1];
      contourCoordinates.push([x, y]);
    }

    // Add the current contour's coordinates to the main array
    edgeCoordinates.push(contourCoordinates);
  }

  // Example usage
  const roundedEdgeCoordinates = roundCoordinates(edgeCoordinates, 1);
  
  // Cleanup
  src.delete();
 
  dst.delete();

  contours.delete();
  hierarchy.delete();
  //#endregion
  callback(); // run callback function

  tf.engine().endScope(); // end of scoping
};

function roundCoordinates(coordinates, precision = 1) {
  return coordinates.map(contour =>
    contour.map(point =>
      point.map(coord =>
        typeof coord === 'number' ? parseFloat(coord.toFixed(precision)) : coord
      )
    )
  );
}
