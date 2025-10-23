import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectTrigger,
  SelectValue,
  SelectGroup,
  SelectItem,
  SelectLabel,
} from "@/components/ui/select";
import { DownloadAvgSpeed } from "@/DownloadData";
import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, XAxis } from "recharts";

export const FuelEff = ({ id }) => {
  // TEST DATA - TO BE CHANGED WITH ORIGINAL DATA
  const [data, setData] = useState([]);
  // download data
  useEffect(() => {
    async function fetchData() {
      const response = await DownloadAvgSpeed(id);
      setData(response);
    }
    fetchData();
  }, []);
  const chartConfig = {
    value: {
      label: "Average Speed (km)",
      color: "#2563eb",
    },
  };
  const [selectType, changeType] = useState("route");
  return (
    <>
      <Card className="h-[45vh]">
        <CardHeader className="grid grid-cols-12">
          <div className="col-span-6 lg:col-span-8 mt-auto mb-auto">
            <h1>Fuel Effeciency</h1>
          </div>
          <div className="block md:flex lg:flex col-span-6 lg:col-span-4">
            {/* <div className="block max-w-[100%]"></div> */}
            <div className="block mt-auto mb-auto me-5">
              <h1>Sort by</h1>
            </div>
            <Select onValueChange={(e) => changeType(e)}>
              <SelectTrigger className="max-w-[84%]">
                <SelectValue placeholder="Route" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectItem value="route">Route</SelectItem>
                  <SelectItem value="avg_speed">Average Speed</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[30vh] w-full">
            <LineChart accessibilityLayer data={data}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="time"
                tickLine={true}
                label={{ value: "Day", position: "insideBottom", offset: 0 }}
                axisLine={true}
                tickFormatter={(value) => value.slice(0, 3)}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Line dataKey="value" fill="var(--color-desktop)" radius={4} />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </>
  );
};
