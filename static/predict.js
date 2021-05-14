const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

let model;
(async function () {
	try
    {
		model = await tf.loadLayersModel("file://model.json");
     	//model = await tf.loadModel('https://saketchaturvedi.github.io/tfjs-models/model.json');	
    $('.progress-bar').hide();
	}catch(error){
		console.error(error);
	}
})();

$("#predict-button").click(async function () {
	
	let image = $('#selected-image').get(0);
	
	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([224,224])
    .toFloat();

	let offset = tf.scalar(127.5);
	
	tensor = tensor.sub(offset)
	.div(offset)
	.expandDims();
	
	let predictions = await model.predict(tensor).data();
	let top3 = Array.from(predictions)
		.map(function (p, i) { 
			return {
				probability: p,
				className: SKIN_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
	
	
    $("#prediction-list").empty();
    top3.forEach(function (p) {

	    $("#prediction-list").append(`<li> chance d'être ${p.className} est: ${p.probability.toFixed(3)*100} %</li>`);

	
	});	
});
