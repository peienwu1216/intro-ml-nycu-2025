import { motion } from 'framer-motion';

export default function Hero() {
  return (
    <section className="flex flex-col items-center lg:items-start justify-center text-center lg:text-left">
      <div className="max-w-2xl">
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="font-serif text-5xl md:text-7xl lg:text-8xl text-black mb-6 leading-[0.9]"
        >
          Rate Your <br/>
          <span className="italic font-light">Composition</span>
        </motion.h1>
        
        <motion.div 
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 1, delay: 0.5, ease: "circOut" }}
          className="w-24 h-[1px] bg-black my-8 mx-auto lg:mx-0"
        />

        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="font-sans text-sm md:text-base text-neutral-500 tracking-wide max-w-md leading-relaxed mx-auto lg:mx-0"
        >
          Elevate your photography through the lens of Machine Learning.
          Analyze balance, symmetry, and rule of thirds to perfect your craft.
        </motion.p>
      </div>
    </section>
  );
}
