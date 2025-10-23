import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { CartesianGrid, Pie, PieChart } from "recharts";

export const TruckDirection = () => {
  const chartData = [
    { position: "north", data: 275, fill: "var(--color-n)" },
    { position: "south", data: 201, fill: "var(--color-s)" },
    { position: "east", data: 187, fill: "var(--color-e)" },
    { position: "west", data: 173, fill: "var(--color-w)" },
  ];
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
          <h1>Truck Direction</h1>
        </div>
      </CardHeader>
      <CardContent className="mt-auto mb-auto">
        <ChartContainer
          config={chartConfig}
          className="[&_.recharts-pie-label-text]:fill-foreground mx-auto max-h-[300px]  pb-0"
        >
          <PieChart className="h-[10vh]">
            <ChartTooltip content={<ChartTooltipContent hideLabel />} />
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
