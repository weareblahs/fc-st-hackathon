import ky from "ky";

export const DownloadAvgSpeed = async (id) => {
  const data = await ky.get(`http://127.0.0.1:5000/${id}/avg_speed`).json();
  return data;
};

export const DownloadTopDrivers = async (id) => {
  const data = await ky.get(`http://127.0.0.1:5000/${id}/top_drivers`).json();
  return data;
};

export const DownloadDirection = async (id) => {
  const data = await ky
    .get(`http://127.0.0.1:5000/${id}/direction_analysis`)
    .json();
  return data;
};

export const DownloadTruckList = async (id) => {
  const data = await ky
    .get(`http://127.0.0.1:5000/${id}/list_of_trucks`)
    .text();
  return data;
};

export const DownloadTruckPos = async (id, truck) => {
  const data = await ky.get(`http://127.0.0.1:5000/${id}/${truck}/pos`).text();
  return data;
};

export const DownloadTruckStats = async (id, truck) => {
  const data = await ky
    .get(`http://127.0.0.1:5000/${id}/${truck}/stats`)
    .json();
  return data;
};

export const predict = async (id, truck) => {
  const data = await ky
    .get(`http://127.0.0.1:5000/${id}/${truck}/data_for_predict`)
    .json();
  if (data) {
    const prediction = await ky
      .post("http://localhost:8000/predict", {
        json: data,
      })
      .json();
    console.log(
      Math.round(prediction["predicted_route_efficiency"] * 100 * 100) / 100
    );
  } else {
    return 0;
  }
};
