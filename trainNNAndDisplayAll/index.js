let model;
const keysAllowed = ['c','d','e','f'];
let currentPoints = [];
let predictions = [];
let trainModelButton;
let resolutionSlider;
let runPredictionMapButton;
let isTrained = false;
let predictionResolution = 25;

function setup(){
    createCanvas(700, 700);
    trainModelButton = createButton('Train');
    runPredictionMapButton = createButton('Predict');
    createP('Prediction resolution')
    resolutionSlider = createSlider(10, 100, 50);

    trainModelButton.mousePressed(() => {
        model.normalizeData();
        model.train({
            epochs: 200
        }, ()=>{
            isTrained = true
            predictAll();
        });
    });

    runPredictionMapButton.mousePressed(() => {
        predictions = [];
        predictAll();
    });

    let mlOptions = {
        inputs: ['x', 'y'],
        outputs: ['label'],
        task: 'classification',
        debug: true,
    };

    model = ml5.neuralNetwork(mlOptions);
}

function keyPressed(){
    if(keysAllowed.indexOf(key)+1){
        currentPoints.push({
            x: mouseX,
            y: mouseY,
            label: key.toUpperCase()
        });
        model.addData({x: mouseX, y: mouseY}, {targets: key.toUpperCase()})
    }
}

function predictAll(){
    predictionResolution = resolutionSlider.value();
    for(let x = 0; x < 750/predictionResolution; x++){
        for(let y = 0; y < 750/predictionResolution; y++){
            model.classify({x: x*predictionResolution, y: y*predictionResolution}, (err, result) => {
                predictions.push({
                    x: x * predictionResolution,
                    y: y * predictionResolution,
                    label: result[0].label
                });
            })
        }
    }
}

function mousePressed(){
    if(isTrained){
        let x = mouseX;
        let y = mouseY;
        model.classify({x: x, y: y}, (err, result)=> {
            if(err) return;
            currentPoints.push({x: x, y: y, label: result[0].label, color: [0, 0, 255, 100]})
        });
    }
}

function drawPoint(xPos, yPos, label, background){
    stroke(0);
    if(!background){
        noFill()
    }else{
        fill(...background)
    }
    ellipse(xPos, yPos, 24);
    fill(0);
    noStroke();
    textAlign(CENTER, CENTER);
    text(label, xPos, yPos);
}

function drawRect(x, y, label){
    stroke(0);
    noFill();
    rect(x,y,predictionResolution,predictionResolution);
    fill(0);
    noStroke();
    text(label, x - (predictionResolution /2) , y - (predictionResolution/2))
}

function draw(){
    background(200);
    currentPoints.forEach(pointObj => {
        drawPoint(pointObj.x, pointObj.y, pointObj.label, pointObj.color);
    });
    predictions.forEach(pointObj => {
        drawRect(pointObj.x, pointObj.y, pointObj.label);
    });
}