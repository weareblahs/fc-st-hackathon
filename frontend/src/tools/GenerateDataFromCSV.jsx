export const GenerateDataFromCSV = (d) => {
  const form = new FormData();
  const csvFile =
    d instanceof Blob || d instanceof File
      ? d
      : new Blob([d], { type: "text/csv" });
  form.append("file", csvFile, "data.csv");
  return import("ky").then(({ default: ky }) =>
    ky.post("http://localhost:5000/upload", { body: form })
  );
};
