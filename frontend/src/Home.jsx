import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";
import "leaflet/dist/leaflet.css";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Bar, BarChart, CartesianGrid, XAxis } from "recharts";
import { Card, CardContent } from "./components/ui/card";
import { FuelEff } from "./parts/FuelEff";
import { TopEffDrivers } from "./parts/TopEffDrivers";
import { TruckDirection } from "./parts/TruckDirection";
import { Button } from "./components/ui/button";
import { GenerateDataFromCSV } from "./tools/GenerateDataFromCSV";
import { Location } from "./parts/Location";
import { Prediction } from "./parts/Prediction";

function Home() {
  const [uuid, setid] = useState("extracted_data");
  const navigate = useNavigate();

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const isCsvMime = file.type === "text/csv";
    const isCsvExt = file.name.toLowerCase().endsWith(".csv");

    if (!isCsvMime && !isCsvExt) {
      alert("Please upload a CSV file.");
      e.target.value = "";
      return;
    }

    try {
      const text = await file.text();
      if (text != "") {
        GenerateDataFromCSV(text);
      }
    } catch (err) {
      alert(err.message);
    } finally {
      e.target.value = "";
    }
  };

  return (
    <div className="p-6 dark">
      <div className="grid grid-cols-2 pb-3 text-white">
        <div className="mt-auto mb-auto">
          <h1 className="">Dashboard</h1>
        </div>
        <div className="flex ms-auto">
          <h1 className="pe-4 mt-auto mb-auto">Want a report?</h1>
          {/* <input
            id="file-input"
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={handleFileChange}
          />
          <Button asChild>
            <label htmlFor="file-input" className="cursor-pointer">
              Upload a CSV file
            </label>
          </Button> */}
          <Button onClick={() => navigate(`/dataReport/${uuid}`)}>
            Get data report
          </Button>
        </div>
      </div>
      <div className="h-[45vh]">
        <FuelEff id={uuid} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-4">
        <div className="grid col-span-1 lg:col-span-2 mt-4 me-0 lg: me-6">
          <div>
            <TopEffDrivers id={uuid} />
          </div>
        </div>
        <div className="grid col-span-1 lg:col-span-2 mt-4">
          <div>
            <TruckDirection id={uuid} />
          </div>
        </div>
      </div>{" "}
      <div className="mt-4 w-full">
        <Location id={uuid} />
      </div>
      <div className="mt-4 w-full">
        <Prediction id={uuid} />
      </div>
    </div>
  );
}

export default Home;
