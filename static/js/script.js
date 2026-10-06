async function analyzeSentiment(){

let text=document.getElementById("review").value;

if(text.trim()==""){

alert("Please enter some text!");

return;

}

const response=await fetch("/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({text:text})

});

const data=await response.json();

document.getElementById("emoji").innerHTML=data.emoji;

document.getElementById("prediction").innerHTML=data.prediction;

document.getElementById("confidence").innerHTML="Confidence : "+data.confidence+" %";

document.getElementById("bar").style.width=data.confidence+"%";

}

function clearText(){

document.getElementById("review").value="";

document.getElementById("emoji").innerHTML="😊";

document.getElementById("prediction").innerHTML="Waiting...";

document.getElementById("confidence").innerHTML="Confidence : 0%";

document.getElementById("bar").style.width="0%";

}