/* update bound parameters and variable attributes (start, nominal, min, max) */
#include "BatteryCell_model.h"
#if defined(__cplusplus)
extern "C" {
#endif


/*
equation index: 16
type: SIMPLE_ASSIGN
$START.T_cell = T_env
*/
static void BatteryCell_eqFunction_16(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,16};
  (data->modelData->realVarsData[1] /* T_cell STATE(1) */).attribute .start = (data->simulationInfo->realParameter[3] /* T_env PARAM */);
    (data->localData[0]->realVars[1] /* T_cell STATE(1) */) = (data->modelData->realVarsData[1] /* T_cell STATE(1) */).attribute .start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[1].info /* T_cell */.name, (modelica_real) (data->localData[0]->realVars[1] /* T_cell STATE(1) */));
  TRACE_POP
}

/*
equation index: 17
type: SIMPLE_ASSIGN
$START.SOC = SOC_init
*/
static void BatteryCell_eqFunction_17(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,17};
  (data->modelData->realVarsData[0] /* SOC STATE(1) */).attribute .start = (data->simulationInfo->realParameter[2] /* SOC_init PARAM */);
    (data->localData[0]->realVars[0] /* SOC STATE(1) */) = (data->modelData->realVarsData[0] /* SOC STATE(1) */).attribute .start;
    infoStreamPrint(LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[0].info /* SOC */.name, (modelica_real) (data->localData[0]->realVars[0] /* SOC STATE(1) */));
  TRACE_POP
}
OMC_DISABLE_OPT
int BatteryCell_updateBoundVariableAttributes(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  /* min ******************************************************** */
  infoStreamPrint(LOG_INIT, 1, "updating min-values");
  if (ACTIVE_STREAM(LOG_INIT)) messageClose(LOG_INIT);
  
  /* max ******************************************************** */
  infoStreamPrint(LOG_INIT, 1, "updating max-values");
  if (ACTIVE_STREAM(LOG_INIT)) messageClose(LOG_INIT);
  
  /* nominal **************************************************** */
  infoStreamPrint(LOG_INIT, 1, "updating nominal-values");
  if (ACTIVE_STREAM(LOG_INIT)) messageClose(LOG_INIT);
  
  /* start ****************************************************** */
  infoStreamPrint(LOG_INIT, 1, "updating primary start-values");
  BatteryCell_eqFunction_16(data, threadData);

  BatteryCell_eqFunction_17(data, threadData);
  if (ACTIVE_STREAM(LOG_INIT)) messageClose(LOG_INIT);
  
  TRACE_POP
  return 0;
}

OMC_DISABLE_OPT
int BatteryCell_updateBoundParameters(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  TRACE_POP
  return 0;
}

#if defined(__cplusplus)
}
#endif

