/* Linearization */
#include "BatteryCell_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
const char *BatteryCell_linear_model_frame()
{
  return "model linearized_model \"BatteryCell\"\n"
  "  parameter Integer n = 3 \"number of states\";\n"
  "  parameter Integer m = 1 \"number of inputs\";\n"
  "  parameter Integer p = 0 \"number of outputs\";\n"
  "  parameter Real x0[n] = %s;\n"
  "  parameter Real u0[m] = %s;\n"
  "\n"
  "  parameter Real A[n, n] =\n\t[%s];\n\n"
  "  parameter Real B[n, m] =\n\t[%s];\n\n"
  "  parameter Real C[p, n] = zeros(p, n);%s\n\n"
  "  parameter Real D[p, m] = zeros(p, m);%s\n\n"
  "\n"
  "  Real x[n](start=x0);\n"
  "  input Real u[m](start=u0);\n"
  "  output Real y[p];\n"
  "\n"
  "  Real 'x_SOC' = x[1];\n"
  "  Real 'x_T_cell' = x[2];\n"
  "  Real 'x_V_p1' = x[3];\n"
  "  Real 'u_I_load' = u[1];\n"
  "equation\n"
  "  der(x) = A * x + B * u;\n"
  "  y = C * x + D * u;\n"
  "end linearized_model;\n";
}
const char *BatteryCell_linear_model_datarecovery_frame()
{
  return "model linearized_model \"BatteryCell\"\n"
  "  parameter Integer n = 3 \"number of states\";\n"
  "  parameter Integer m = 1 \"number of inputs\";\n"
  "  parameter Integer p = 0 \"number of outputs\";\n"
  "  parameter Integer nz = 4 \"data recovery variables\";\n"
  "  parameter Real x0[3] = %s;\n"
  "  parameter Real u0[1] = %s;\n"
  "  parameter Real z0[4] = %s;\n"
  "\n"
  "  parameter Real A[n, n] =\n\t[%s];\n\n"
  "  parameter Real B[n, m] =\n\t[%s];\n\n"
  "  parameter Real C[p, n] = zeros(p, n);%s\n\n"
  "  parameter Real D[p, m] = zeros(p, m);%s\n\n"
  "  parameter Real Cz[nz, n] =\n\t[%s];\n\n"
  "  parameter Real Dz[nz, m] =\n\t[%s];\n\n"
  "\n"
  "  Real x[n](start=x0);\n"
  "  input Real u[m](start=u0);\n"
  "  output Real y[p];\n"
  "  output Real z[nz];\n"
  "\n"
  "  Real 'x_SOC' = x[1];\n"
  "  Real 'x_T_cell' = x[2];\n"
  "  Real 'x_V_p1' = x[3];\n"
  "  Real 'u_I_load' = u[1];\n"
  "  Real 'z_I_load' = z[1];\n"
  "  Real 'z_V_ocv' = z[2];\n"
  "  Real 'z_V_terminal' = z[3];\n"
  "  Real 'z_heat_gen' = z[4];\n"
  "equation\n"
  "  der(x) = A * x + B * u;\n"
  "  y = C * x + D * u;\n"
  "  z = Cz * x + Dz * u;\n"
  "end linearized_model;\n";
}
#if defined(__cplusplus)
}
#endif

