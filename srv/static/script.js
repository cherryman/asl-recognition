let video = undefined;
let canvas = undefined;
let results = undefined;
let ctx = undefined;
let url = undefined;

function startVideo() {
  return navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
      video.play();
    })
    .catch((e) => {
      console.error(e);
      throw e;
    });
}

const stripRe = /^data:image\/png;base64,/;
function snap() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  let data = canvas.toDataURL("image/png");
  return data.replace(stripRe, "");
}

function send() {
  const data = snap();
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: data,
      width: video.videoWidth,
      height: video.videoHeight,
    }),
  })
    .then((r) => r.json())
    .then((r) => {
      results.innerText = r.data.join(", ");
    })
    .catch((e) => console.error(e));
}

function init(reqUrl) {
  video = document.getElementById("video");
  canvas = document.getElementById("canvas");
  results = document.getElementById("results");
  ctx = canvas.getContext("2d");
  url = reqUrl;

  startVideo();
  window.setInterval(send, 2000);
}
