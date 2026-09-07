import { Router, type IRouter } from "express";
import healthRouter from "./health";
import recognitionRouter from "./recognition";
import trainingRouter from "./training";
import evaluationRouter from "./evaluation";

const router: IRouter = Router();

router.use(healthRouter);
router.use(recognitionRouter);
router.use(trainingRouter);
router.use(evaluationRouter);

export default router;
