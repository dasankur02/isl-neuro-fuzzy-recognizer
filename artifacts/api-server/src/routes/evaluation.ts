import { Router, type IRouter } from "express";
import { GetEvaluationSummaryResponse } from "@workspace/api-zod";
import { getEvaluationSummary } from "../lib/isl-model";

const router: IRouter = Router();

router.get("/evaluation/summary", (_req, res) => {
  res.json(GetEvaluationSummaryResponse.parse(getEvaluationSummary()));
});

export default router;