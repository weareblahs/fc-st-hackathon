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
import { useEffect, useState } from "react";
import { DownloadTopDrivers } from "@/DownloadData";

export const TopEffDrivers = ({ id }) => {
  const [drivers, topDrivers] = useState([]);

  useEffect(() => {
    async function fetchData() {
      const response = await DownloadTopDrivers(id);
      topDrivers(response);
    }
    fetchData();
  }, []);
  // src: https://stackoverflow.com/questions/4687723/how-to-convert-minutes-to-hours-minutes-and-add-various-time-values-together-usi
  const convertMinsToHrsMins = (mins) => {
    let h = Math.floor(mins / 60);
    let m = mins % 60;
    h = h < 10 ? "0" + h : h; // (or alternatively) h = String(h).padStart(2, '0')
    m = m < 10 ? "0" + m : m; // (or alternatively) m = String(m).padStart(2, '0')
    return `${h}h ${m}m`;
  };
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
                <TableHead className="w-[20%]">Total moving time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {drivers.length == 0 ? (
                <h1>No data</h1>
              ) : (
                drivers.map((d) => {
                  return (
                    <TopEffDriversCard
                      rank={d.rank + 1}
                      plate={d.vehicle}
                      distance={convertMinsToHrsMins(Math.round(d.time))}
                    />
                  );
                })
              )}
            </TableBody>
          </Table>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};
