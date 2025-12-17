import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import AboutModal from './AboutModal';

interface HeaderProps {
  onHomeClick?: () => void;
}

export default function Header({ onHomeClick }: HeaderProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);

  const menuItems = ['HOME', 'ABOUT'];

  return (
    <>
      <AboutModal isOpen={isAboutOpen} onClose={() => setIsAboutOpen(false)} />
      
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 md:px-8 py-6 bg-white/80 backdrop-blur-sm"
      >
        <button 
          onClick={onHomeClick}
          className="flex flex-col items-start hover:opacity-70 transition-opacity z-50 relative"
        >
          <span className="font-serif text-xl font-bold tracking-tighter text-black">AESTHETIC</span>
          <span className="font-sans text-[0.6rem] tracking-[0.3em] text-neutral-500 ml-1">IMAGE RATING</span>
        </button>
        
        {/* Desktop Nav */}
        <nav className="hidden md:block">
          <ul className="flex gap-8">
            {menuItems.map((item, i) => (
              <motion.li 
                key={item}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + i * 0.1 }}
              >
                <a 
                  href="#" 
                  onClick={(e) => {
                    e.preventDefault();
                    if (item === 'HOME' && onHomeClick) {
                      onHomeClick();
                    } else if (item === 'ABOUT') {
                      setIsAboutOpen(true);
                    }
                  }}
                  className="relative font-sans text-xs font-medium tracking-widest text-black group"
                >
                  {item}
                  <span className="absolute -bottom-2 left-0 w-0 h-[1px] bg-black transition-all duration-300 group-hover:w-full" />
                </a>
              </motion.li>
            ))}
          </ul>
        </nav>

        {/* Mobile Burger Button */}
        <button 
          className="md:hidden z-50 p-2 -mr-2"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </motion.header>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-white flex flex-col items-center justify-center md:hidden"
          >
            <nav>
              <ul className="flex flex-col items-center gap-8">
                {menuItems.map((item, i) => (
                  <motion.li
                    key={item}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 20 }}
                    transition={{ delay: 0.1 + i * 0.1 }}
                  >
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        setIsMenuOpen(false);
                        if (item === 'HOME' && onHomeClick) {
                          onHomeClick();
                        } else if (item === 'ABOUT') {
                          setIsAboutOpen(true);
                        }
                      }}
                      className="font-serif text-3xl text-black hover:italic transition-all"
                    >
                      {item}
                    </a>
                  </motion.li>
                ))}
              </ul>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
