let x_vals = [];
let y_vals = [];

let isClickingMouse = false;

const width = 700;
const height = 700;
const model = tf.sequential();
const hidden = tf.layers.dense({
    units: 4,
    inputShape: [3],
    activation: 'sigmoid',

});
const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
});

model.add(hidden);
model.add(output);

model.compile({
    optimizer: tf.train.sgd(0.5),
    loss: tf.losses.meanSquaredError
});

async function setup(){
    createCanvas(width, height);
    let loss = 1.1;
    const minLoss = 0.0001;
    let loops = 0;
    let trainingIn = tf.tensor2d([
        [0,0,1],
        [0,1,1],
        [0,1,0],
    ]);
    let trainingOut = tf.tensor2d([
        [1],
        [1],
        [0],
    ]);
    while( loss > minLoss){
        loops++;
        await model.fit(trainingIn, trainingOut,{
                epochs: loops * 5,
                shuffle: true,
            }
        ).then((result) => {
            tf.dispose(this);
            loss = result.history.loss[0];
            console.log(loss)
        });
    }
    tf.dispose([trainingOut, trainingIn]);
    let output = model.predict(tf.tensor2d([
        [1,1,0]
    ]));
    output.print();
    output = model.predict(tf.tensor2d([
        [1,1,1]
    ]));
    output.print();
    output = model.predict(tf.tensor2d([
        [0,0,0]
    ]));
    output.print();
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
