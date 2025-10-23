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
