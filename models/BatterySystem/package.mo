package BatterySystem
  "Industrial Grade Battery Digital Twin Library"
  
  // 引用同目录下的其他组件
  model BatteryCell end BatteryCell;
  model CoolingSystem end CoolingSystem;
  
  annotation(
    version = "2.0",
    author = "Digital Twin Architect",
    date = "2023-10-27"
  );
end BatterySystem;