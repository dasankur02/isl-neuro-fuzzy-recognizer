import { Router, type IRouter } from "express";
import {
  ClassifyRecognitionBody,
  ClassifyRecognitionResponse,
  GetRecognitionDemoResponse,
} from "@workspace/api-zod";
import { classifyLandmarks, makeDemo } from "../lib/isl-model";

const router: IRouter = Router();

router.get("/recognition/demo", (_req, res) => {
  res.json(GetRecognitionDemoResponse.parse(makeDemo()));
});

router.post("/recognition/classify", (req, res) => {
  const parsed = ClassifyRecognitionBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Expected exactly 21 hand landmarks." });
    return;
  }
  res.json(ClassifyRecognitionResponse.parse(classifyLandmarks(
    parsed.data.landmarks,
    parsed.data.sequence,
  )));
});

export default router;