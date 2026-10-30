import type { Model } from '../data/models'

/** Parameter counts by component, derived from the architecture. Norms and biases are ignored (<0.1%). */
export interface ParamBreakdown {
  /** Input embedding + output head (one matrix if tied) */
  embed: number
  /** Output head alone (read every decode step; equals the embedding when tied) */
  lmHead: number
  attention: number
  /** Dense FFN, or MoE router for MoE models */
  dense: number
  /** All MoE expert weights */
  experts: number
  /** Expert weights touched per token */
  activeExperts: number
  total: number
  /** Params used per token (embedding row lookup excluded from embed, counted as lm head) */
  active: number
}

export function paramBreakdown(m: Model): ParamBreakdown {
  const embedMatrix = m.vocab * m.hidden
  const embed = embedMatrix * (m.tiedEmbeddings ? 1 : 2)
  const lmHead = embedMatrix
  const attnPerLayer = 2 * m.hidden * m.heads * m.headDim + 2 * m.hidden * m.kvHeads * m.headDim
  const attention = attnPerLayer * m.layers

  let dense = 0
  let experts = 0
  let activeExperts = 0
  if (m.moe) {
    const perExpert = 3 * m.hidden * m.moe.expertIntermediate
    dense = m.hidden * m.moe.experts * m.layers
    experts = perExpert * m.moe.experts * m.layers
    activeExperts = perExpert * m.moe.activeExperts * m.layers
  } else {
    dense = 3 * m.hidden * m.intermediate * m.layers
  }

  const total = embed + attention + dense + experts
  const active = lmHead + attention + dense + activeExperts
  return { embed, lmHead, attention, dense, experts, activeExperts, total, active }
}
