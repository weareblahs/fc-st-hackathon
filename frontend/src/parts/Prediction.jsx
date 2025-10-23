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
  useEffect(() => {
    async function fetchData() {
      if (truck != "") {
        const response = await predict(id, truck);
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
      <Card className="flex flex-col">
        <CardContent className="flex-1 p-0 overflow-hidden">
          <div className="grid grid-cols-12">
            <div className="col-span-4 ms-5 h-full">
              <div className="ms-2">
                <h1 className="text-8xl font-bold mb-4">Data Prediction</h1>
                <h1 className="text-2xl mb-4">
                  Predict the route effeciency for the next trip.
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
            <div className="col-span-8"></div>
          </div>
        </CardContent>
      </Card>
    </>
  );
};
