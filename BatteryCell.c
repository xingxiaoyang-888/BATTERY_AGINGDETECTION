/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "BatteryCell_model.h"
#include "simulation/solver/events.h"

/* FIXME these defines are ugly and hard to read, why not use direct function pointers instead? */
#define prefixedName_performSimulation BatteryCell_performSimulation
#define prefixedName_updateContinuousSystem BatteryCell_updateContinuousSystem
#include <simulation/solver/perform_simulation.c.inc>

#define prefixedName_performQSSSimulation BatteryCell_performQSSSimulation
#include <simulation/solver/perform_qss_simulation.c.inc>


/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int BatteryCell_input_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  (data->localData[0]->realVars[6] /* I_load variable */) = data->simulationInfo->inputVars[0];
  
  TRACE_POP
  return 0;
}

int BatteryCell_input_function_init(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->inputVars[0] = data->modelData->realVarsData[6].attribute.start;
  
  TRACE_POP
  return 0;
}

int BatteryCell_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->modelData->realVarsData[6].attribute.start = data->simulationInfo->inputVars[0];
  
  TRACE_POP
  return 0;
}

int BatteryCell_inputNames(DATA *data, char ** names){
  TRACE_PUSH

  names[0] = (char *) data->modelData->realVarsData[6].info.name;
  
  TRACE_POP
  return 0;
}

int BatteryCell_data_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  TRACE_POP
  return 0;
}

int BatteryCell_dataReconciliationInputNames(DATA *data, char ** names){
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int BatteryCell_dataReconciliationUnmeasuredVariables(DATA *data, char ** names)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int BatteryCell_output_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int BatteryCell_setc_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int BatteryCell_setb_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}


/*
equation index: 10
type: SIMPLE_ASSIGN
heat_gen = I_load ^ 2.0 * R_internal + V_p1 * I_load
*/
void BatteryCell_eqFunction_10(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,10};
  modelica_real tmp0;
  tmp0 = (data->localData[0]->realVars[6] /* I_load variable */);
  (data->localData[0]->realVars[9] /* heat_gen variable */) = ((tmp0 * tmp0)) * ((data->simulationInfo->realParameter[1] /* R_internal PARAM */)) + ((data->localData[0]->realVars[2] /* V_p1 STATE(1) */)) * ((data->localData[0]->realVars[6] /* I_load variable */));
  TRACE_POP
}
/*
equation index: 11
type: SIMPLE_ASSIGN
$DER.T_cell = (heat_gen + h_conv * (T_env - T_cell)) / C_thermal
*/
void BatteryCell_eqFunction_11(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,11};
  (data->localData[0]->realVars[4] /* der(T_cell) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[9] /* heat_gen variable */) + ((data->simulationInfo->realParameter[5] /* h_conv PARAM */)) * ((data->simulationInfo->realParameter[3] /* T_env PARAM */) - (data->localData[0]->realVars[1] /* T_cell STATE(1) */)),(data->simulationInfo->realParameter[0] /* C_thermal PARAM */),"C_thermal",equationIndexes);
  TRACE_POP
}
/*
equation index: 12
type: SIMPLE_ASSIGN
$DER.V_p1 = 0.0002 * I_load + (-0.05) * V_p1
*/
void BatteryCell_eqFunction_12(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,12};
  (data->localData[0]->realVars[5] /* der(V_p1) STATE_DER */) = (0.0002) * ((data->localData[0]->realVars[6] /* I_load variable */)) + (-0.05) * ((data->localData[0]->realVars[2] /* V_p1 STATE(1) */));
  TRACE_POP
}
/*
equation index: 13
type: SIMPLE_ASSIGN
V_ocv = 3.0 + 0.9 * SOC + 0.3 * SOC ^ 0.5 + (-0.1) * SOC ^ 3.0
*/
void BatteryCell_eqFunction_13(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13};
  modelica_real tmp1;
  modelica_real tmp2;
  tmp1 = (data->localData[0]->realVars[0] /* SOC STATE(1) */);
  if(!(tmp1 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(SOC) was %g should be >= 0", tmp1);
    }
  }tmp2 = (data->localData[0]->realVars[0] /* SOC STATE(1) */);
  (data->localData[0]->realVars[7] /* V_ocv variable */) = 3.0 + (0.9) * ((data->localData[0]->realVars[0] /* SOC STATE(1) */)) + (0.3) * (sqrt(tmp1)) + (-0.1) * ((tmp2 * tmp2 * tmp2));
  TRACE_POP
}
/*
equation index: 14
type: SIMPLE_ASSIGN
V_terminal = V_ocv + (-I_load) * R_internal - V_p1
*/
void BatteryCell_eqFunction_14(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14};
  (data->localData[0]->realVars[8] /* V_terminal variable */) = (data->localData[0]->realVars[7] /* V_ocv variable */) + ((-(data->localData[0]->realVars[6] /* I_load variable */))) * ((data->simulationInfo->realParameter[1] /* R_internal PARAM */)) - (data->localData[0]->realVars[2] /* V_p1 STATE(1) */);
  TRACE_POP
}
/*
equation index: 15
type: SIMPLE_ASSIGN
$DER.SOC = (-0.0002777777777777778) * I_load / capacity_Ah
*/
void BatteryCell_eqFunction_15(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,15};
  (data->localData[0]->realVars[3] /* der(SOC) STATE_DER */) = DIVISION_SIM((-0.0002777777777777778) * ((data->localData[0]->realVars[6] /* I_load variable */)),(data->simulationInfo->realParameter[4] /* capacity_Ah PARAM */),"capacity_Ah",equationIndexes);
  TRACE_POP
}

OMC_DISABLE_OPT
int BatteryCell_functionDAE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  int equationIndexes[1] = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  BatteryCell_functionLocalKnownVars(data, threadData);
  BatteryCell_eqFunction_10(data, threadData);

  BatteryCell_eqFunction_11(data, threadData);

  BatteryCell_eqFunction_12(data, threadData);

  BatteryCell_eqFunction_13(data, threadData);

  BatteryCell_eqFunction_14(data, threadData);

  BatteryCell_eqFunction_15(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  TRACE_POP
  return 0;
}


int BatteryCell_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}


/* forwarded equations */
extern void BatteryCell_eqFunction_10(DATA* data, threadData_t *threadData);
extern void BatteryCell_eqFunction_11(DATA* data, threadData_t *threadData);
extern void BatteryCell_eqFunction_12(DATA* data, threadData_t *threadData);
extern void BatteryCell_eqFunction_15(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  {
    BatteryCell_eqFunction_10(data, threadData);
    threadData->lastEquationSolved = 10;
  }
  {
    BatteryCell_eqFunction_11(data, threadData);
    threadData->lastEquationSolved = 11;
  }
  {
    BatteryCell_eqFunction_12(data, threadData);
    threadData->lastEquationSolved = 12;
  }
  {
    BatteryCell_eqFunction_15(data, threadData);
    threadData->lastEquationSolved = 15;
  }
}

int BatteryCell_functionODE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  BatteryCell_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  TRACE_POP
  return 0;
}

/* forward the main in the simulation runtime */
extern int _main_SimulationRuntime(int argc, char**argv, DATA *data, threadData_t *threadData);

#include "BatteryCell_12jac.h"
#include "BatteryCell_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks BatteryCell_callback = {
   (int (*)(DATA *, threadData_t *, void *)) BatteryCell_performSimulation,    /* performSimulation */
   (int (*)(DATA *, threadData_t *, void *)) BatteryCell_performQSSSimulation,    /* performQSSSimulation */
   BatteryCell_updateContinuousSystem,    /* updateContinuousSystem */
   BatteryCell_callExternalObjectDestructors,    /* callExternalObjectDestructors */
   NULL,    /* initialNonLinearSystem */
   NULL,    /* initialLinearSystem */
   NULL,    /* initialMixedSystem */
   #if !defined(OMC_NO_STATESELECTION)
   BatteryCell_initializeStateSets,
   #else
   NULL,
   #endif    /* initializeStateSets */
   BatteryCell_initializeDAEmodeData,
   BatteryCell_functionODE,
   BatteryCell_functionAlgebraics,
   BatteryCell_functionDAE,
   BatteryCell_functionLocalKnownVars,
   BatteryCell_input_function,
   BatteryCell_input_function_init,
   BatteryCell_input_function_updateStartValues,
   BatteryCell_data_function,
   BatteryCell_output_function,
   BatteryCell_setc_function,
   BatteryCell_setb_function,
   BatteryCell_function_storeDelayed,
   BatteryCell_function_storeSpatialDistribution,
   BatteryCell_function_initSpatialDistribution,
   BatteryCell_updateBoundVariableAttributes,
   BatteryCell_functionInitialEquations,
   1, /* useHomotopy - 0: local homotopy (equidistant lambda), 1: global homotopy (equidistant lambda), 2: new global homotopy approach (adaptive lambda), 3: new local homotopy approach (adaptive lambda)*/
   NULL,
   BatteryCell_functionRemovedInitialEquations,
   BatteryCell_updateBoundParameters,
   BatteryCell_checkForAsserts,
   BatteryCell_function_ZeroCrossingsEquations,
   BatteryCell_function_ZeroCrossings,
   BatteryCell_function_updateRelations,
   BatteryCell_zeroCrossingDescription,
   BatteryCell_relationDescription,
   BatteryCell_function_initSample,
   BatteryCell_INDEX_JAC_A,
   BatteryCell_INDEX_JAC_B,
   BatteryCell_INDEX_JAC_C,
   BatteryCell_INDEX_JAC_D,
   BatteryCell_INDEX_JAC_F,
   BatteryCell_INDEX_JAC_H,
   BatteryCell_initialAnalyticJacobianA,
   BatteryCell_initialAnalyticJacobianB,
   BatteryCell_initialAnalyticJacobianC,
   BatteryCell_initialAnalyticJacobianD,
   BatteryCell_initialAnalyticJacobianF,
   BatteryCell_initialAnalyticJacobianH,
   BatteryCell_functionJacA_column,
   BatteryCell_functionJacB_column,
   BatteryCell_functionJacC_column,
   BatteryCell_functionJacD_column,
   BatteryCell_functionJacF_column,
   BatteryCell_functionJacH_column,
   BatteryCell_linear_model_frame,
   BatteryCell_linear_model_datarecovery_frame,
   BatteryCell_mayer,
   BatteryCell_lagrange,
   BatteryCell_pickUpBoundsForInputsInOptimization,
   BatteryCell_setInputData,
   BatteryCell_getTimeGrid,
   BatteryCell_symbolicInlineSystem,
   BatteryCell_function_initSynchronous,
   BatteryCell_function_updateSynchronous,
   BatteryCell_function_equationsSynchronous,
   BatteryCell_inputNames,
   BatteryCell_dataReconciliationInputNames,
   BatteryCell_dataReconciliationUnmeasuredVariables,
   NULL,
   NULL,
   NULL,
   -1,
   NULL,
   NULL,
   -1

};

#define _OMC_LIT_RESOURCE_0_name_data "BatteryCell"
#define _OMC_LIT_RESOURCE_0_dir_data "F:/Battery_Desktop/models/openmodelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_name,11,_OMC_LIT_RESOURCE_0_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir,38,_OMC_LIT_RESOURCE_0_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,2,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir)}};
void BatteryCell_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &BatteryCell_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "BatteryCell";
  data->modelData->modelFilePrefix = "BatteryCell";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "F:/Battery_Desktop/models/openmodelica";
  data->modelData->modelGUID = "{a6c834fe-2d31-4283-9703-211720f9570c}";
  data->modelData->encrypted = 0;
  #if defined(OPENMODELICA_XML_FROM_FILE_AT_RUNTIME)
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  #else
  #if defined(_MSC_VER) /* handle joke compilers */
  {
  /* for MSVC we encode a string like char x[] = {'a', 'b', 'c', '\0'} */
  /* because the string constant limit is 65535 bytes */
  static const char contents_init[] =
    #include "BatteryCell_init.c"
    ;
  static const char contents_info[] =
    #include "BatteryCell_info.c"
    ;
    data->modelData->initXMLData = contents_init;
    data->modelData->modelDataXml.infoXMLData = contents_info;
  }
  #else /* handle real compilers */
  data->modelData->initXMLData =
  #include "BatteryCell_init.c"
    ;
  data->modelData->modelDataXml.infoXMLData =
  #include "BatteryCell_info.c"
    ;
  #endif /* defined(_MSC_VER) */
  #endif /* defined(OPENMODELICA_XML_FROM_FILE_AT_RUNTIME) */
  data->modelData->modelDataXml.fileName = "BatteryCell_info.json";
  data->modelData->resourcesDir = NULL;
  data->modelData->runTestsuite = 0;
  data->modelData->nStates = 3;
  data->modelData->nVariablesReal = 10;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesInteger = 0;
  data->modelData->nVariablesBoolean = 0;
  data->modelData->nVariablesString = 0;
  data->modelData->nParametersReal = 6;
  data->modelData->nParametersInteger = 0;
  data->modelData->nParametersBoolean = 0;
  data->modelData->nParametersString = 0;
  data->modelData->nInputVars = 1;
  data->modelData->nOutputVars = 0;
  data->modelData->nAliasReal = 0;
  data->modelData->nAliasInteger = 0;
  data->modelData->nAliasBoolean = 0;
  data->modelData->nAliasString = 0;
  data->modelData->nZeroCrossings = 0;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 0;
  data->modelData->nMathEvents = 0;
  data->modelData->nExtObjs = 0;
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 0;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 18;
  data->modelData->nMixedSystems = 0;
  data->modelData->nLinearSystems = 0;
  data->modelData->nNonLinearSystems = 0;
  data->modelData->nStateSets = 0;
  data->modelData->nJacobians = 6;
  data->modelData->nOptimizeConstraints = 0;
  data->modelData->nOptimizeFinalConstraints = 0;
  data->modelData->nDelayExpressions = 0;
  data->modelData->nBaseClocks = 0;
  data->modelData->nSpatialDistributions = 0;
  data->modelData->nSensitivityVars = 0;
  data->modelData->nSensitivityParamVars = 0;
  data->modelData->nSetcVars = 0;
  data->modelData->ndataReconVars = 0;
  data->modelData->nSetbVars = 0;
  data->modelData->nRelatedBoundaryConditions = 0;
  data->modelData->linearizationDumpLanguage = OMC_LINEARIZE_DUMP_LANGUAGE_MODELICA;
}

static int rml_execution_failed()
{
  fflush(NULL);
  fprintf(stderr, "Execution failed!\n");
  fflush(NULL);
  return 1;
}


#if defined(__MINGW32__) || defined(_MSC_VER)

#if !defined(_UNICODE)
#define _UNICODE
#endif
#if !defined(UNICODE)
#define UNICODE
#endif

#include <windows.h>
char** omc_fixWindowsArgv(int argc, wchar_t **wargv)
{
  char** newargv;
  /* Support for non-ASCII characters
  * Read the unicode command line arguments and translate it to char*
  */
  newargv = (char**)malloc(argc*sizeof(char*));
  for (int i = 0; i < argc; i++) {
    newargv[i] = omc_wchar_to_multibyte_str(wargv[i]);
  }
  return newargv;
}

#define OMC_MAIN wmain
#define OMC_CHAR wchar_t
#define OMC_EXPORT __declspec(dllexport) extern

#else
#define omc_fixWindowsArgv(N, A) (A)
#define OMC_MAIN main
#define OMC_CHAR char
#define OMC_EXPORT extern
#endif

#if defined(threadData)
#undef threadData
#endif
/* call the simulation runtime main from our main! */
#if defined(OMC_DLL_MAIN_DEFINE)
OMC_EXPORT int omcDllMain(int argc, OMC_CHAR **argv)
#else
int OMC_MAIN(int argc, OMC_CHAR** argv)
#endif
{
  char** newargv = omc_fixWindowsArgv(argc, argv);
  /*
    Set the error functions to be used for simulation.
    The default value for them is 'functions' version. Change it here to 'simulation' versions
  */
  omc_assert = omc_assert_simulation;
  omc_assert_withEquationIndexes = omc_assert_simulation_withEquationIndexes;

  omc_assert_warning_withEquationIndexes = omc_assert_warning_simulation_withEquationIndexes;
  omc_assert_warning = omc_assert_warning_simulation;
  omc_terminate = omc_terminate_simulation;
  omc_throw = omc_throw_simulation;

  int res;
  DATA data;
  MODEL_DATA modelData;
  SIMULATION_INFO simInfo;
  data.modelData = &modelData;
  data.simulationInfo = &simInfo;
  measure_time_flag = 0;
  compiledInDAEMode = 0;
  compiledWithSymSolver = 0;
  MMC_INIT(0);
  omc_alloc_interface.init();
  {
    MMC_TRY_TOP()
  
    MMC_TRY_STACK()
  
    BatteryCell_setupDataStruc(&data, threadData);
    res = _main_initRuntimeAndSimulation(argc, newargv, &data, threadData);
    if(res == 0) {
      res = _main_SimulationRuntime(argc, newargv, &data, threadData);
    }
    
    MMC_ELSE()
    rml_execution_failed();
    fprintf(stderr, "Stack overflow detected and was not caught.\nSend us a bug report at https://trac.openmodelica.org/OpenModelica/newticket\n    Include the following trace:\n");
    printStacktraceMessages();
    fflush(NULL);
    return 1;
    MMC_CATCH_STACK()
    
    MMC_CATCH_TOP(return rml_execution_failed());
  }

  fflush(NULL);
#if !defined(OMC_DLL_MAIN_DEFINE) /* do not exit, return in DLL mode */
  EXIT(res);
#endif
  return res;
}

#ifdef __cplusplus
}
#endif


