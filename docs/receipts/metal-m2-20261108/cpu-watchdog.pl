use strict;
use warnings;
use POSIX qw(WNOHANG);
$| = 1;
my $child = fork();
die "fork: $!" unless defined $child;
if ($child == 0) { exec @ARGV; die "exec: $!"; }
my $start = time;
my $reason = '';
while (1) {
  my $done = waitpid($child, WNOHANG);
  if ($done == $child) { my $status = $?; print STDERR "CPU watchdog child_status=$status elapsed=" . (time-$start) . "s\n"; exit(($status & 127) ? 128+($status & 127) : $status >> 8); }
  my $rss = `ps -p $child -o rss=`;
  $rss =~ s/\s+//g;
  my $pressure = `memory_pressure`;
  my ($free) = $pressure =~ /System-wide memory free percentage: (\d+)%/;
  print STDERR "CPU watchdog elapsed=" . (time-$start) . "s rss_kib=$rss memory_free_percent=" . (defined $free ? $free : 'unknown') . "\n";
  $reason = '900 second deadline' if time-$start >= 900;
  $reason = '17 GiB resident cap' if $rss =~ /^\d+$/ && $rss > 17*1024*1024;
  $reason = 'system free memory below 12 percent' if defined $free && $free < 12;
  if ($reason ne '') { print STDERR "CPU watchdog STOP: $reason\n"; kill 'TERM', $child; sleep 2; kill 'KILL', $child if waitpid($child, WNOHANG) == 0; waitpid($child, 0); exit 124; }
  sleep 5;
}
