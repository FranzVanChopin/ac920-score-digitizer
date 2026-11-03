/* Smooth static UI font. Renderer code MIT; glyph data SIL OFL1.1.
 * See STUDIO_FONT_OFL.txt and STUDIO_FONT_SOURCE.txt. */
#ifndef STUDIO_FONT_H
#define STUDIO_FONT_H
#include <stddef.h>
#include <stdint.h>
/* BGR framebuffer; x,y are the capital-height top-left. Scales1..4 map to
 * 12/18/24/34px. ASCII32..126, proportional advance; other bytes are skipped.
 * Bounds/stride/address overflow are checked; caller supplies stride*h bytes. */
void studio_font_text(uint8_t *dst,unsigned w,unsigned h,size_t stride,
                      unsigned x,unsigned y,const char *text,unsigned scale,uint32_t rgb);
unsigned studio_font_measure(const char *text,unsigned scale);
#endif
