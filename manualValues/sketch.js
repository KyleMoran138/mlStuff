let x_vals = [];
let y_vals = [];

let isClickingMouse = false;

const width = 700;
const height = 700;
let m, b, a, c, learningRate, optimizer;

function setup(){
    createCanvas(width, height);
    m = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    a = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    learningRate = 0.43;
    optimizer = tf.train.adam(learningRate);
}

function mousePressed(){
    isClickingMouse = true;
}

function mouseReleased(){
    isClickingMouse = false;
}

function draw(){
    background(0);
    drawAxis();
    addPointToArrayIfClicking();
    drawPoints();
    if(x_vals.length){
        minimize();
        drawPrediction();
    }
}

function drawAxis(){
    stroke(255);
    strokeWeight(4);
    line(0, height/2, width, height/2);
    line(width/2, 0, width/2, height)
}

function addPointToArrayIfClicking(){
    if(isClickingMouse){
        let {x, y} = normalize(mouseX, mouseY);
        x_vals.push(x);
        y_vals.push(y);
    }
}

function drawPoints(){
    stroke(255);
    strokeWeight(8);
    for(let i = 0; i < x_vals.length; i++){
        let {x, y} = weirdize(x_vals[i], y_vals[i]);
        point(x,y);
    }
}

function normalize(xPos, yPos){
    return {
        x: map(xPos, 0, width, -1, 1),
        y: map(yPos, 0, height, 1, -1)
    }
}

function weirdize(xPos, yPos){
    return {
        x: map(xPos, -1, 1, 0, width),
        y: map(yPos, 1, -1, 0, height)
    }
}

function drawPrediction(){
    tf.tidy(() => {
        const curveX = [];
        for (let x = -1; x < 1.1; x+= 0.05){
            curveX.push(x);
        }
        let curveY = predict(curveX).dataSync();
        beginShape();
        noFill();
        stroke(255);
        strokeWeight(2);
        for(let i = 0; i < curveX.length; i++){
            let {x, y} = weirdize(curveX[i], curveY[i]);
            vertex(x, y);
        }
        endShape();
    });
    textSize(32);
    text("A: "+ (Math.round(a.dataSync() * 100) / 100), (width / 4) * 0, height-10);
    text("B: "+ (Math.round(b.dataSync() * 100) / 100), (width / 4) * 1, height-10);
    text("C: "+ (Math.round(c.dataSync() * 100) / 100), (width / 4) * 2, height-10);
    text("M: "+ (Math.round(m.dataSync() * 100) / 100), (width / 4) * 3, height-10);
    text("X: "+ (Math.round(normalize(mouseX, 0).x * 100) / 100), (width / 4) * 0, height-60);
    text("Y: "+ (Math.round(normalize(0, mouseY).y * 100) / 100), (width / 4) * 1, height-60);
}

function predict(xs){
    const xsTensor = tf.tensor1d(xs);
    const ys = xsTensor.pow(tf.tensor1d([3])).mul(a).add(xsTensor.square().mul(b)).add(xsTensor.mul(c)).add(m);
    return ys;
}

function loss(predictions, labels){
    return predictions.sub(labels).square().mean();
}

function minimize(){
    optimizer.minimize(() => loss(predict(x_vals), tf.tensor1d(y_vals)));
}