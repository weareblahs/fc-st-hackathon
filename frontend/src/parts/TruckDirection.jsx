import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { DownloadDirection } from "@/DownloadData";
import { useEffect, useState } from "react";
import { Pie, PieChart } from "recharts";

export const TruckDirection = ({ id }) => {
  const [chartData, setData] = useState([]);

  useEffect(() => {
    async function fetchData() {
      const response = await DownloadDirection(id);
      setData(response);
    }
    fetchData();
  }, []);

  const chartConfig = {
    n: {
      label: "North",
      color: "var(--chart-1)",
    },
    s: {
      label: "South",
      color: "var(--chart-2)",
    },
    e: {
      label: "East",
      color: "var(--chart-3)",
    },
    w: {
      label: "West",
      color: "var(--chart-4)",
    },
  };
  return (
    <Card className="h-full">
      <CardHeader className="">
        <div className="">
          <h1>Truck Direction by trip count</h1>
        </div>
      </CardHeader>
      <CardContent className="mt-auto mb-auto">
        <ChartContainer
          config={chartConfig}
          className="[&_.recharts-pie-label-text]:fill-foreground mx-auto max-h-[300px]  pb-0"
        >
          <PieChart className="h-[10vh]">
            <ChartTooltip content={<ChartTooltipContent label="test" />} />
            <Pie
              data={chartData}
              dataKey="data"
              nameKey="position"
              label={({ name }) => name.charAt(0).toUpperCase() + name.slice(1)}
            />
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
};
