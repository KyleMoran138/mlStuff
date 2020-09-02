const width = 500;
const height = 400;
const knnClassifier = ml5.KNNClassifier();
let capture, nameInput, captureImage, 
saveModelButton, modelFileInput;
let label = 'Needs to be trained'
let featureExtractor;
let hasExamples = false;

function setup() {
  createCanvas(width, height+50);
  capture = createCapture(VIDEO);
  capture.size(width, height);
  capture.hide();

  createSpan('\nClass name:');
  nameInput = createInput('');
  captureImageButton = createButton('Sample');
  saveModelButton = createButton('Save model');
  modelFileInput = createButton('Load model')

  captureImageButton.mousePressed(captureClassImage);
  saveModelButton.mousePressed(saveModel);
  modelFileInput.mousePressed(loadModelintoKnn);

  featureExtractor = ml5.featureExtractor("MobileNet", ()=>{
    doPredict();
  });
}

function loadModelintoKnn(){

    knnClassifier.load("model.json", () => {
        hasExamples = true;
        doPredict();
    });
}

function saveModel(){
    knnClassifier.save('model.json');
}

function captureClassImage(){
    const features = featureExtractor.infer(capture);
    knnClassifier.addExample(features, nameInput.value());
    if(!hasExamples){
        hasExamples = true;
        doPredict();
    }
}

function doPredict(){
    if(hasExamples){
        const features = featureExtractor.infer(capture);
        knnClassifier.classify(features, (err, result) => {
            label = result.label
            doPredict();
        });
    }
}

function draw() {
  background(0);
  image(capture, 0, 0, width, height);

  textSize(32);
  fill(255)
  text(label, 25, height+35);
}
