/* Initialization */
#include "BatteryCell_model.h"
#include "BatteryCell_11mix.h"
#include "BatteryCell_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void BatteryCell_functionInitialEquations_0(DATA *data, threadData_t *threadData);
extern void BatteryCell_eqFunction_15(DATA *data, threadData_t *threadData);


/*
equation index: 2
type: SIMPLE_ASSIGN
SOC = SOC_init
*/
void BatteryCell_eqFunction_2(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2};
  (data->localData[0]->realVars[0] /* SOC STATE(1) */) = (data->simulationInfo->realParameter[2] /* SOC_init PARAM */);
  TRACE_POP
}
extern void BatteryCell_eqFunction_13(DATA *data, threadData_t *threadData);


/*
equation index: 4
type: SIMPLE_ASSIGN
T_cell = T_env
*/
void BatteryCell_eqFunction_4(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4};
  (data->localData[0]->realVars[1] /* T_cell STATE(1) */) = (data->simulationInfo->realParameter[3] /* T_env PARAM */);
  TRACE_POP
}

/*
equation index: 5
type: SIMPLE_ASSIGN
V_p1 = $START.V_p1
*/
void BatteryCell_eqFunction_5(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5};
  (data->localData[0]->realVars[2] /* V_p1 STATE(1) */) = (data->modelData->realVarsData[2] /* V_p1 STATE(1) */).attribute .start;
  TRACE_POP
}
extern void BatteryCell_eqFunction_10(DATA *data, threadData_t *threadData);

extern void BatteryCell_eqFunction_11(DATA *data, threadData_t *threadData);

extern void BatteryCell_eqFunction_14(DATA *data, threadData_t *threadData);

extern void BatteryCell_eqFunction_12(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void BatteryCell_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  BatteryCell_eqFunction_15(data, threadData);
  BatteryCell_eqFunction_2(data, threadData);
  BatteryCell_eqFunction_13(data, threadData);
  BatteryCell_eqFunction_4(data, threadData);
  BatteryCell_eqFunction_5(data, threadData);
  BatteryCell_eqFunction_10(data, threadData);
  BatteryCell_eqFunction_11(data, threadData);
  BatteryCell_eqFunction_14(data, threadData);
  BatteryCell_eqFunction_12(data, threadData);
  TRACE_POP
}

int BatteryCell_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->discreteCall = 1;
  BatteryCell_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  TRACE_POP
  return 0;
}

/* No BatteryCell_functionInitialEquations_lambda0 function */

int BatteryCell_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int *equationIndexes = NULL;
  double res = 0.0;

  
  TRACE_POP
  return 0;
}


#if defined(__cplusplus)
}
#endif

