import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./App.css";
import "leaflet/dist/leaflet.css";
import Home from "./Home";
import { PrintData } from "./subpages/PrintData";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dataReport/:id" element={<PrintData />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
