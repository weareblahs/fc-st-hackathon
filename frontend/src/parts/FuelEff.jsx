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
      // You can await here
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
            <h1>Average Speed</h1>
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
