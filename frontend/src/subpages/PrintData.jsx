import { Button } from "@/components/ui/button";
import { DownloadImageURLs } from "@/DownloadData";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

export const PrintData = () => {
  const { id } = useParams();
  const [img, setimg] = useState([]);
  useEffect(() => {
    async function fetchData() {
      if (img != []) {
        const response = await DownloadImageURLs(id);
        setimg(response);
      }
    }
    fetchData();
  }, []);
  const dataInfo = [
    "Pearson Correlation Matrix (Lower Triangle)",
    "Pearson Correlation Matrix (Lower Triangle)",
    "Route Efficiency average by Hour of Day",
    "Route Efficiency Distribution plot and Box plot",
    "Route Efficiency compared to Days of the Week (Box Plot)",
    "Route Efficiency during peak hours (Box Plot)",
    "Speed compared to Route Efficiency Average",
    "Comparison between Route Efficiency (Median vs Average)",
    "Comparison between Route Efficiency (Standard Deviation vs Average)",
    "Comparison between Route Efficiency (Average)",
    "Comparison between Average Speed and Average Route Efficiency",
    "Comparison between Maximum Speed and Average Route Efficiency",
    "Comparison between Speed and Average Route Efficiency",
    "Comparison between Speed (Standard Deviation) and Average Route Efficiency",
    "Comparison between Stops per kilometer and Average Route Efficiency",
    "Comparison between Trip Distance and Average Route Efficiency",
    "How many trips are done in each day of the week?",
    "How many trips are done in peak hours?",
    "How many trips are done during weekends?",
    "How many trips are done in the days of the week?",
    "Speed density",
    "Route Efficiency (median) density",
    "Route Efficiency (standard deviation) density",
    "Route Efficiency density",
    "Speed (average) density",
    "Speed (maximum) density",
    "Speed density",
    "Speed (standard deviation) density",
    "Stops per kilometer density",
    "Trip distance density",
  ];
  return (
    <div className="text-white">
      {img != [] ? (
        <center>
          <div className="p-8">
            <div className="grid grid-cols-2 print:grid-cols-1">
              <div className="text-start print:text-center print:text-black">
                <h1 className="text-2xl">Data Report for ID: {id}</h1>
                <h1 className="text-base print:hidden">
                  Click on the images to view a larger version of the data.{" "}
                </h1>
              </div>
              <div className="ms-auto me-5 print:hidden">
                <Button onClick={() => window.print()}>Print</Button>
              </div>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 print:grid-cols-1">
              {img.images?.map((i, ind) => {
                return (
                  <div className="mt-auto mb-auto p-2 print:text-black">
                    <a href={`/api/${i}`}>
                      <img src={`/api/${i}`} className="w-full" />
                    </a>
                    <p>{dataInfo[ind]}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </center>
      ) : (
        <h1>Loading data...</h1>
      )}
    </div>
  );
};
