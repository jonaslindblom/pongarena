import { Trophy } from 'lucide-react'
import './RankBadge.css'

export function RankBadge({ rank, label = 'Rank' }: { rank: number | null; label?: string }) {
  const medal = rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : undefined
  const description = rank === null ? label + ' pending' : label + ' ' + rank
  return <span className="rank-badge" data-rank={rank ?? undefined} data-medal={medal}
    role="img" aria-label={description} title={description}>
    {medal && <Trophy size={16} strokeWidth={1.8} fill="currentColor" fillOpacity={0.18} aria-hidden="true" />}
    <span aria-hidden="true">{rank === null ? '—' : '#' + rank}</span>
  </span>
}
