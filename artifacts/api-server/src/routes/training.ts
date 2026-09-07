import { Router, type IRouter } from "express";
import { RunTrainingBody, RunTrainingResponse } from "@workspace/api-zod";
import { makeTrainingResult } from "../lib/isl-model";

const router: IRouter = Router();

router.post("/training/run", (req, res) => {
  const parsed = RunTrainingBody.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({
      error: "Training configuration is invalid. Upload at least 10 labeled samples covering all five digits.",
    });
    return;
  }
  const labels = new Set(parsed.data.dataset.map((sample) => sample.label));
  const counts = parsed.data.dataset.reduce<Record<string, number>>((result, sample) => {
    result[sample.label] = (result[sample.label] ?? 0) + 1;
    return result;
  }, {});
  if (labels.size !== 5 || Object.values(counts).some((count) => count < 2)) {
    res.status(400).json({
      error: "Training needs at least two labeled landmark samples for each of ONE through FIVE.",
    });
    return;
  }
  const epochs = Math.round(parsed.data.epochs ?? 250);
  try {
    res.json(
      RunTrainingResponse.parse(
        makeTrainingResult(epochs, parsed.data.augmentation ?? true, parsed.data.dataset),
      ),
    );
  } catch (error) {
    res.status(400).json({
      error: error instanceof Error ? error.message : "The uploaded dataset could not be trained.",
    });
  }
});

export default router;