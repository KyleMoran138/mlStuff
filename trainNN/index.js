let model;
const keysAllowed = ['c','d','e','f'];
let currentPoints = [];
let trainModelButton;
let isTrained = false;

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