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
import { useState } from "react";
import { CartesianGrid, Line, LineChart, XAxis } from "recharts";

export const FuelEff = () => {
  // TEST DATA - TO BE CHANGED WITH ORIGINAL DATA
  const testData = [
    { month: "January", desktop: 186, mobile: 80 },
    { month: "February", desktop: 305, mobile: 200 },
    { month: "March", desktop: 237, mobile: 120 },
    { month: "April", desktop: 73, mobile: 190 },
    { month: "May", desktop: 209, mobile: 130 },
    { month: "June", desktop: 214, mobile: 140 },
  ];
  const chartConfig = {
    desktop: {
      label: "Desktop",
      color: "#2563eb",
    },
    mobile: {
      label: "Mobile",
      color: "#60a5fa",
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
            <LineChart accessibilityLayer data={testData}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="month"
                tickLine={false}
                tickMargin={10}
                axisLine={false}
                tickFormatter={(value) => value.slice(0, 3)}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Line dataKey="desktop" fill="var(--color-desktop)" radius={4} />
              <Line dataKey="mobile" fill="var(--color-mobile)" radius={4} />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </>
  );
};
