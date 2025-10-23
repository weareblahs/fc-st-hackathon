import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TopEffDriversCard } from "./TopEffDrivers/card";
import { ScrollArea } from "@/components/ui/scroll-area";

export const TopEffDrivers = () => {
  return (
    <Card>
      <CardHeader className="grid grid-cols-12">
        <div className="col-span-6 lg:col-span-8 mt-auto mb-auto">
          <h1>Top Efficiency Drivers</h1>
        </div>
        <div className="block md:flex lg:flex col-span-6 lg:col-span-4">
          {/* <div className="block max-w-[100%]"></div> */}
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[35vh]">
          <Table className="">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[10%]">#</TableHead>
                <TableHead className="w-[70%]">License plate</TableHead>
                <TableHead className="w-[20%]">Total distance (km)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
              <TopEffDriversCard rank="1" plate="1" distance="1" />
            </TableBody>
          </Table>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};
