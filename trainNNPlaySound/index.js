let model;
const keysAllowed = ['c','d','e','f'];
let currentPoints = [];
let trainModelButton;
let isTrained = false;
let noteEnvelope;
let wave;

function setup(){
    createCanvas(700, 700);
    trainModelButton = createButton('train');
    trainModelButton.mousePressed(() => {
        model.normalizeData();
        model.train({
            epochs: 200
        }, ()=>{isTrained = true});
    });

    let mlOptions = {
        inputs: ['x', 'y'],
        outputs: ['label'],
        task: 'classification',
        debug: true,
    };

    noteEnvelope = p5.Envelope();
    noteEnvelope.setADSR(0.05, 0.1, 0.5, 1);
    noteEnvelope.setRange(1.2, 0);

    wave = p5.Oscillator();
    wave.setType('square');
    wave.start();
    wave.freq(440);
    wave.amp(noteEnvelope);

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

function mousePressed(){
    if(isTrained){
        let x = mouseX;
        let y = mouseY;
        model.classify({x: x, y: y}, (err, result)=> {
            if(err) return;
            currentPoints.push({x: x, y: y, label: result[0].label, color: [0, 0, 255, 100]})
            getAudioContext().resume();
            noteEnvelope.play();
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

function draw(){
    background(200);
    currentPoints.forEach(pointObj => {
        drawPoint(pointObj.x, pointObj.y, pointObj.label, pointObj.color);
    });
}