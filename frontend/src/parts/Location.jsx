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
} from "@/DownloadData";
import { useEffect, useState } from "react";

import {
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  Polyline,
} from "react-leaflet";

// Function to calculate distance between two coordinates
const calculateDistance = (point1, point2) => {
  const [lat1, lon1] = point1;
  const [lat2, lon2] = point2;
  return Math.sqrt(Math.pow(lat2 - lat1, 2) + Math.pow(lon2 - lon1, 2));
};

// Function to smooth coordinates by sorting them in a logical path
// Uses nearest neighbor algorithm to minimize jumps between points
// Also filters coordinates to only include those within bounds:
// Latitude: 1°N to 7°N, Longitude: 100°E to 119°E
const smoothCoordinates = (coordinates) => {
  // Filter coordinates to only include those within the specified bounds
  const filteredCoordinates = coordinates.filter(([lat, lon]) => {
    return lat >= 1 && lat <= 7 && lon >= 100 && lon <= 119;
  });

  if (filteredCoordinates.length <= 1) return filteredCoordinates;

  // Maximum distance threshold - if points are farther than this, skip the connection
  // Approximately 0.5 degrees (~55km) - adjust this value as needed
  const MAX_DISTANCE_THRESHOLD = 1;

  const sortedPath = [filteredCoordinates[0]];
  const remaining = [...filteredCoordinates.slice(1)];

  while (remaining.length > 0) {
    const lastPoint = sortedPath[sortedPath.length - 1];
    let nearestIndex = 0;
    let minDistance = calculateDistance(lastPoint, remaining[0]);

    // Find the nearest unvisited point
    for (let i = 1; i < remaining.length; i++) {
      const distance = calculateDistance(lastPoint, remaining[i]);
      if (distance < minDistance) {
        minDistance = distance;
        nearestIndex = i;
      }
    }

    // Only add the point if it's within the reasonable distance threshold
    if (minDistance <= MAX_DISTANCE_THRESHOLD) {
      sortedPath.push(remaining[nearestIndex]);
    }

    // Remove from remaining regardless to avoid infinite loop
    remaining.splice(nearestIndex, 1);
  }

  return sortedPath;
};

export const Location = ({ id }) => {
  const [trucks, selectTrucks] = useState([]);
  const [truck, selectTruck] = useState("");
  const [pos, setpos] = useState([]);
  const [smoothedPos, setSmoothedPos] = useState([]);
  const [truckStats, setStats] = useState([]);
  // Smooth the coordinates when pos changes
  useEffect(() => {
    const smoothed = smoothCoordinates(pos);
    setSmoothedPos(smoothed);
  }, [pos]);

  useEffect(() => {
    async function fetchData() {
      if (truck != "") {
        const response = await DownloadTruckPos(id, truck);
        setpos(JSON.parse(response));
        const response2 = await DownloadTruckStats(id, truck);
        setStats(response2);
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
  // src: https://stackoverflow.com/questions/4687723/how-to-convert-minutes-to-hours-minutes-and-add-various-time-values-together-usi
  const convertMinsToHrsMins = (mins) => {
    let h = Math.floor(mins / 60);
    let m = mins % 60;
    h = h < 10 ? "0" + h : h; // (or alternatively) h = String(h).padStart(2, '0')
    m = m < 10 ? "0" + m : m; // (or alternatively) m = String(m).padStart(2, '0')
    return `${h}h ${m}m`;
  };
  return (
    <>
      <Card className="h-max lg:h-[45vh] flex flex-col">
        <CardContent className="flex-1 p-0 overflow-hidden">
          <div className="grid grid-cols-12 h-full">
            <div className="col-span-12 lg:col-span-4 p-4 ms-8 me-8">
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
              {truck != "" ? (
                <div className="grid grid-cols-2 h-full">
                  <div className="py-6 mt-auto mb-auto">
                    <h1 className="text-center text-4xl font-bold">
                      {truckStats["month_total_distance"]
                        ? `${truckStats["month_total_distance"]}km`
                        : "..."}
                    </h1>
                    <h1 className="text-center text-xl">driven this month</h1>
                  </div>
                  <div className="py-6 mt-auto mb-auto">
                    <h1 className="text-center text-4xl font-bold">
                      {truckStats["day_total_distance_avg"]
                        ? `${truckStats["day_total_distance_avg"]}km`
                        : "..."}
                    </h1>
                    <h1 className="text-center text-xl">
                      driven each day (average)
                    </h1>
                  </div>{" "}
                  <div className="py-6 mt-auto mb-auto">
                    <h1 className="text-center text-4xl font-bold">
                      {" "}
                      {truckStats["total_wasted"]
                        ? `RM${truckStats["total_wasted"]}`
                        : "..."}
                    </h1>
                    <h1 className="text-center text-xl">
                      {truckStats["fuel_liters"]
                        ? `(${truckStats["fuel_liters"]}l)`
                        : "..."}
                      <br />
                      fuel wasted for this month
                    </h1>
                  </div>{" "}
                  <div className="py-6 mt-auto mb-auto">
                    <h1 className="text-center text-4xl font-bold">
                      {" "}
                      {truckStats["total_idle_min"]
                        ? convertMinsToHrsMins(truckStats["total_idle_min"])
                        : "..."}
                    </h1>
                    <h1 className="text-center text-xl">idle for this month</h1>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full mt-2">
                  <h1 className="">
                    Select a truck above to view truck statistics, including
                    total driven distance, fuel statistics, map visualization
                    and others.
                  </h1>
                </div>
              )}
            </div>
            <div className="col-span-12 lg:col-span-8 h-[50vw] lg:h-full w-[96%]">
              {smoothedPos.length > 0 && (
                <MapContainer
                  center={smoothedPos[0]}
                  zoom={7}
                  scrollWheelZoom={true}
                  className="h-full w-full"
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <Polyline positions={smoothedPos} color="blue" />
                </MapContainer>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  );
};
