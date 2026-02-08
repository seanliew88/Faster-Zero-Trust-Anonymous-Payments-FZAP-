import TransferForm from "@/components/TransferForm";
import NetworkStatus from "@/components/NetworkStatus";

const Index = () => {
  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center bg-background bg-grid px-4 py-12 overflow-hidden">
      {/* Ambient glow blobs */}
      <div className="pointer-events-none absolute top-1/4 -left-32 w-96 h-96 rounded-full bg-primary/5 blur-3xl" />
      <div className="pointer-events-none absolute bottom-1/4 -right-32 w-96 h-96 rounded-full bg-accent/5 blur-3xl" />

      {/* Header */}
      <div className="text-center mb-10 space-y-3 relative z-10">
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-gradient-primary">
          Quantum Swap
        </h1>
        <p className="text-sm sm:text-base text-muted-foreground max-w-md mx-auto leading-relaxed">
          Anonymous USDC transfers via QAOA-optimised cross-chain mixing.
          Untraceable. Verified by Flare Data Connector.
        </p>
      </div>

      {/* Form */}
      <div className="relative z-10 w-full">
        <TransferForm />
      </div>

      {/* Network Status */}
      <div className="mt-10 relative z-10">
        <NetworkStatus />
      </div>
    </div>
  );
};

export default Index;
