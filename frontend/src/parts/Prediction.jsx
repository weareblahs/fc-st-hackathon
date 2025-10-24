import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DownloadTruckList,
  DownloadTruckPos,
  DownloadTruckStats,
  predict,
} from "@/DownloadData";
import { useEffect, useState } from "react";

import {
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  Polyline,
} from "react-leaflet";

export const Prediction = ({ id }) => {
  const [trucks, selectTrucks] = useState([]);
  const [truck, selectTruck] = useState("");
  const [predictResult, setPredictResult] = useState(-1);
  useEffect(() => {
    async function fetchData() {
      if (truck != "") {
        const response = await predict(id, truck);
        setPredictResult(response);
      }
    }
    fetchData();
  }, [truck]);
  useEffect(() => {
    async function fetchData() {
      const response = await DownloadTruckList(id);
      selectTrucks(JSON.parse(response));
    }
    fetchData();
  }, [id]);

  return (
    <>
      <Card className="hidden md:flex lg:flex flex-col">
        <CardContent className="flex-1 p-0 overflow-hidden">
          <div className="grid grid-cols-12">
            <div className="col-span-4 ms-5 h-full">
              <div className="ms-2">
                <h1 className="text-8xl font-bold mb-4">Data Prediction</h1>
                <h1 className="text-2xl mb-4">
                  Predict the vehicle effeciency for the next trip.
                </h1>
              </div>
              <Select onValueChange={(v) => selectTruck(v)}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a truck" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {trucks.map((t) => {
                      return (
                        <SelectItem key={t} value={t}>
                          {t}
                        </SelectItem>
                      );
                    })}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
            <div className="col-span-8 ms-auto me-auto mt-auto mb-auto">
              <h1 className="text-9xl font-bold">
                {predictResult != -1 ? `${predictResult}%` : "..."}
              </h1>
              <h1 className="text-2xl text-center">
                effecient route for next trip
              </h1>
            </div>
          </div>
        </CardContent>
      </Card>
      <Card className="flex md:hidden lg:hidden flex-col">
        <CardContent className="flex-1 p-0 overflow-hidden">
          <div className="grid grid-cols-12">
            <div className="col-span-12  ms-8 h-full">
              <div className="ms-0">
                <h1 className="text-4xl lg:text-8xl font-bold mb-4">
                  Data Prediction
                </h1>
                <h1 className="text-base lg:text-2xl mb-4">
                  Predict the vehicle effeciency for the next trip.
                </h1>
              </div>
              <div className="pe-6">
                <Select onValueChange={(v) => selectTruck(v)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select a truck" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {trucks.map((t) => {
                        return (
                          <SelectItem key={t} value={t}>
                            {t}
                          </SelectItem>
                        );
                      })}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <center></center>
            <div className="col-span-12 lg:col-span-8 text-center ms-auto me-auto mt-auto mb-auto">
              {predictResult != -1 ? (
                <>
                  <h1 className="text-7xl font-bold">
                    {predictResult != -1 ? `${predictResult}%` : "..."}
                  </h1>
                  <h1 className="text-xl text-center">
                    effecient route for next trip
                  </h1>
                </>
              ) : (
                <h1 className="hidden md:block lg:block">
                  Select a truck at the left side to begin predicting.
                </h1>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  );
};
