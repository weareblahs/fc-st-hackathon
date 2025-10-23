import { TableCell, TableRow } from "@/components/ui/table";

export const TopEffDriversCard = ({ rank, plate, distance }) => {
  return (
    <>
      <TableRow>
        <TableCell>{rank}</TableCell>
        <TableCell>{plate}</TableCell>
        <TableCell>{distance}</TableCell>
      </TableRow>
    </>
  );
};
